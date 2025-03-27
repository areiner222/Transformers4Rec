#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import optree as tree
import torch
import cupy as cp
import cudf
import pandas as pd
from merlin.models.utils.doc_utils import docstring_parameter
from merlin.schema import Tags, TagsType

from merlin_standard_lib import Schema
from merlin.dataloader.ops.embeddings import EmbeddingOperator
from merlin.core.dispatch import get_lib

from ..block.base import BlockOrModule, BuildableBlock, SequentialBlock
from ..block.mlp import MLPBlock
from ..masking import MaskSequence, masking_registry
from ..tabular.base import (
    TABULAR_MODULE_PARAMS_DOCSTRING,
    AsTabular,
    TabularAggregationType,
    TabularModule,
    TabularTransformationType,
)
from . import embedding
from .tabular import TABULAR_FEATURES_PARAMS_DOCSTRING, TabularFeatures
from ...config.tags import CustomTags
from ...utils.pipeline import extend_categorify_domain

pd_lib = get_lib()
np_lib = cp if pd_lib == cudf else np

@docstring_parameter(
    tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING,
    embedding_features_parameters=embedding.EMBEDDING_FEATURES_PARAMS_DOCSTRING,
)
class SequenceEmbeddingFeatures(embedding.EmbeddingFeatures):
    """Input block for embedding-lookups for categorical features. This module produces 3-D tensors,
    this is useful for sequential models like transformers.

    Parameters
    ----------
    {embedding_features_parameters}
    padding_idx: int
        The symbol to use for padding.
    {tabular_module_parameters}
    """

    def __init__(
        self,
        feature_config: Dict[str, embedding.FeatureConfig],
        item_id: Optional[str] = None,
        padding_idx: int = 0,
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
    ):
        self.padding_idx = padding_idx
        # super(SequenceEmbeddingFeatures, self).__init__(
        super().__init__(
            feature_config=feature_config,
            item_id=item_id,
            pre=pre,
            post=post,
            aggregation=aggregation,
            schema=schema,
        )

    def table_to_embedding_module(self, table: embedding.TableConfig) -> torch.nn.Embedding:
        embedding_table = torch.nn.Embedding(
            table.vocabulary_size, table.dim, padding_idx=self.padding_idx
        )
        if table.initializer is not None:
            table.initializer(embedding_table.weight)
        return embedding_table

    def forward_output_size(self, input_sizes):
        sizes = {}

        for fname, fconfig in self.feature_config.items():
            fshape = input_sizes[fname]
            sizes[fname] = torch.Size(list(fshape) + [fconfig.table.dim])

        return sizes


@docstring_parameter(
    tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING,
    tabular_features_parameters=TABULAR_FEATURES_PARAMS_DOCSTRING,
)
class TabularSequenceFeatures(TabularFeatures):
    """Input module that combines different types of features to a sequence: continuous,
    categorical & text.

    Parameters
    ----------
    {tabular_features_parameters}
    projection_module: BlockOrModule, optional
        Module that's used to project the output of this module, typically done by an MLPBlock.
    masking: MaskSequence, optional
         Masking to apply to the inputs.
    {tabular_module_parameters}

    """

    EMBEDDING_MODULE_CLASS = SequenceEmbeddingFeatures

    def __init__(
        self,
        continuous_module: Optional[TabularModule] = None,
        categorical_module: Optional[TabularModule] = None,
        pretrained_embedding_module: Optional[TabularModule] = None,
        pretrained_embedding_ops_module: Optional[TabularModule] = None,
        projection_module: Optional[BlockOrModule] = None,
        # exclude_item_id_embedding: bool = False,
        masking: Optional[MaskSequence] = None,
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        **kwargs
    ):
        super().__init__(
            continuous_module,
            categorical_module,
            pretrained_embedding_module,
            pretrained_embedding_ops_module,
            pre=pre,
            post=post,
            aggregation=aggregation,
            schema=schema,
            **kwargs
        )
        # self.exclude_item_id_embedding = exclude_item_id_embedding
        self._item_id = None
        self.projection_module = projection_module
        self.set_masking(masking)
        self._item_component_cols_set = False
        self._cached_item_metadata = None
        self.item_filter = None
        
        # we need to ensure that we have at least one column that is an item_component
        # this can come in the form of an embedding op or a categorical op
        # self.item_component_cat_cols = []
        # self.item_component_emb_cols = []

    # def _set_item_component_cols(self):
    #     if not self._item_component_cols_set and self.exclude_item_id_embedding:
    #         self.item_component_cat_cols = self.schema.select_by_tag(CustomTags.ITEM_COMPONENT).column_names
    #         self.item_component_emb_cols = []
    #         if self.pretrained_embedding_ops_module is not None:
    #             item_emb_component_col = self.pretrained_embedding_ops_module.col_map.get(self.item_id)
    #             if item_emb_component_col is not None:
    #                 self.item_component_emb_cols = [item_emb_component_col]
    #         self._item_component_cols_set = True
    
    @property
    def item_component_cols(self):
        if self.schema is None:
            return []
        return self.schema.select_by_tag(CustomTags.ITEM_COMPONENT).column_names
    
    @property
    def content_id_cols(self):
        if self.schema is None:
            return []
        return self.schema.select_by_tag(CustomTags.CONTENT_ID).column_names
    
    @property
    def content_emb_cols(self):
        if self.schema is None or self.pretrained_embedding_ops_module is None:
            return []
        pretrained_emb_inp_cols = list(set(self.pretrained_embedding_ops_module.col_map.keys()))
        content_id_cols = self.schema.select_by_tag(CustomTags.CONTENT_ID).column_names
        for col in pretrained_emb_inp_cols:
            assert col in content_id_cols, f"pretrained_emb_inp_cols must be a subset of content_id_cols, but {col} is not in {content_id_cols}"
        return pretrained_emb_inp_cols
    
    @property
    def item_component_and_content_emb_cols(self):
        return list(set(self.item_component_cols + self.content_emb_cols))
    
    @property
    def item_component_and_content_out_cols(self):
        out_cols = self.item_component_cols
        for col in self.content_emb_cols:
            assert self.pretrained_embedding_ops_module is not None, "pretrained_embedding_ops_module must be set"
            out_cols.append(self.pretrained_embedding_ops_module.col_map[col])
        return out_cols
    
    @property
    def item_component_and_content_out_size(self):
        embeddings = self.categorical_module.embedding_tables
        item_component_sizes = {col: embeddings[col].weight.shape[1] for col in self.item_component_cols}
        content_emb_cols = self.content_emb_cols
        if len(content_emb_cols) > 0:
            embeddings = self.pretrained_embedding_ops_module.embedding_tables
            content_emb_sizes = {col: embeddings[col].weight.shape[1] for col in content_emb_cols}
            item_component_sizes.update(content_emb_sizes)
        return item_component_sizes
            
    @classmethod
    def from_schema(  # type: ignore
        cls,
        schema: Schema,
        continuous_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.CONTINUOUS,),
        categorical_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.CATEGORICAL,),
        pretrained_embeddings_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.EMBEDDING,),
        pretrained_embedding_ops: Optional[List[EmbeddingOperator]] = None,
        exclude_item_id_embedding: bool = False,
        aggregation: Optional[str] = None,
        automatic_build: bool = True,
        max_sequence_length: Optional[int] = None,
        continuous_projection: Optional[Union[List[int], int]] = None,
        continuous_soft_embeddings: bool = False,
        projection: Optional[Union[torch.nn.Module, BuildableBlock]] = None,
        d_output: Optional[int] = None,
        masking: Optional[Union[str, MaskSequence]] = None,
        **kwargs
    ) -> "TabularSequenceFeatures":
        """Instantiates ``TabularFeatures`` from a ``DatasetSchema``

        Parameters
        ----------
        schema : DatasetSchema
            Dataset schema
        continuous_tags : Optional[Union[TagsType, Tuple[Tags]]], optional
            Tags to filter the continuous features, by default Tags.CONTINUOUS
        categorical_tags : Optional[Union[TagsType, Tuple[Tags]]], optional
            Tags to filter the categorical features, by default Tags.CATEGORICAL
        aggregation : Optional[str], optional
            Feature aggregation option, by default None
        automatic_build : bool, optional
            Automatically infers input size from features, by default True
        max_sequence_length : Optional[int], optional
            Maximum sequence length for list features by default None
        continuous_projection : Optional[Union[List[int], int]], optional
            If set, concatenate all numerical features and project them by a number of MLP layers.
            The argument accepts a list with the dimensions of the MLP layers, by default None
        continuous_soft_embeddings : bool
            Indicates if the  soft one-hot encoding technique must be used to represent
            continuous features, by default False
        projection: Optional[Union[torch.nn.Module, BuildableBlock]], optional
            If set, project the aggregated embeddings vectors into hidden dimension vector space,
            by default None
        d_output: Optional[int], optional
            If set, init a MLPBlock as projection module to project embeddings vectors,
            by default None
        masking: Optional[Union[str, MaskSequence]], optional
            If set, Apply masking to the input embeddings and compute masked labels, It requires
            a categorical_module including an item_id column, by default None

        Returns
        -------
        TabularFeatures
            Returns ``TabularFeatures`` from a dataset schema
        """
        # build output
        output: TabularSequenceFeatures = super().from_schema(  # type: ignore
            schema=schema,
            continuous_tags=continuous_tags,
            categorical_tags=categorical_tags,
            pretrained_embeddings_tags=pretrained_embeddings_tags,
            pretrained_embedding_ops=pretrained_embedding_ops,
            exclude_item_id_embedding=exclude_item_id_embedding,
            aggregation=aggregation,
            automatic_build=automatic_build,
            max_sequence_length=max_sequence_length,
            continuous_projection=continuous_projection,
            continuous_soft_embeddings=continuous_soft_embeddings,
            **kwargs
        )
        
        if d_output and projection:
            raise ValueError("You cannot specify both d_output and projection at the same time")
        if (projection or masking or d_output) and not aggregation:
            # TODO: print warning here for clarity
            output.aggregation = "concat"  # type: ignore
        hidden_size = output.output_size()

        if d_output and not projection:
            projection = MLPBlock([d_output])
        if projection and hasattr(projection, "build"):
            projection = projection.build(hidden_size)  # type: ignore
        if projection:
            output.projection_module = projection  # type: ignore
            hidden_size = projection.output_size()  # type: ignore

        if isinstance(masking, str):
            masking = masking_registry.parse(masking)(
                hidden_size=output.output_size()[-1], **kwargs
            )
        # set the item_id
        output._item_id = schema.select_by_tag(Tags.ITEM_ID).first.name if schema.select_by_tag(Tags.ITEM_ID) else None
            
        if masking and not getattr(output, "item_id", None):
            raise ValueError("For masking a categorical_module is required including an item_id.")
        output.set_masking(masking)  # type: ignore
        
        # set schema
        # output._set_item_component_cols()

        return output

    @property
    def masking(self):
        return self._masking

    def set_masking(self, value):
        self._masking = value

    @property
    def item_id(self) -> Optional[str]:
        if self._item_id is not None:
            return self._item_id
        elif "categorical_module" in self.to_merge:
            return getattr(self.to_merge["categorical_module"], "item_id", None)

        return None
    
    def item_ids(self, inputs) -> torch.Tensor:
        return inputs[self.item_id]

    @property
    def item_embedding_table(self) -> Optional[torch.nn.Module]:
        if "categorical_module" in self.to_merge:
            return getattr(self.to_merge["categorical_module"], "item_embedding_table", None)

        return None
    
    # @property
    # def using_item_component_embedding(self):
    #     return len(self.item_component_emb_cols) > 0 or len(self.item_component_cat_cols) > 0
    
    def get_item_metadata(self):
        if self._cached_item_metadata is None:
            item_col = self.schema[self.item_id]
            item_cat_path = item_col.properties['cat_path']
            item_meta = pd.read_parquet(item_cat_path)
            if self.item_filter is not None:
                # Perform inner join on all available columns between item_meta and item_filter
                common_cols = list(set(item_meta.columns) & set(self.item_filter.columns))
                item_meta = item_meta.reset_index().merge(self.item_filter, on=common_cols, how='inner').set_index('index')
                item_meta.index.name = None
            self._cached_item_metadata = item_meta
        return self._cached_item_metadata
    
    def clear_item_metadata_cache(self):
        """Clear the cached item metadata. This will force a fresh load of the metadata on the next call to get_item_metadata."""
        self._cached_item_metadata = None
    
    def get_item_cat_component_metadata(self, inverse=False, _with_start_idx=False):
        comp_meta = {}
        for col in self.item_component_and_content_emb_cols:
            col_s = self.schema[col]
            cat_path = col_s.properties['cat_path']
            meta = pd.read_parquet(cat_path)[col]
            start_idx = meta.index.start
            res = pd.Series(meta.index, index=meta.values) if inverse else meta
            if _with_start_idx:
                res = (res, start_idx)
            comp_meta[col] = res
        return comp_meta
    
    def build_universe_input(self, item_metadata=None):
        """
        NOTE:
        we assume that the item_metadata is monotonically indexed up to the max domain
        """
        if item_metadata is not None:
            for col in self.item_component_cols:
                if col not in item_metadata:
                    raise ValueError(f"item_metadata must contain all item_component_cat_cols, but {col} is missing")
        else:
            item_metadata = self.get_item_metadata()
        # inp_idx_init = np_lib.arange(item_metadata.index.start)
        # item_metadata = item_metadata if item_metadata is not None else item_metadata_base
        # Note: if we are using a custom item_metadata, it must be true that any custom embedding ops
        # have been updated to reflect the new item_metadata size and index!
        # inp = {
        #     self.item_id: np.arange(self.schema[self.item_id].int_domain.max + 1)
        # }
        inp = {
            self.item_id: np.concatenate([np.arange(3), item_metadata.index.values])
        }
        meta = self.get_item_cat_component_metadata(inverse=True, _with_start_idx=True)
        for col in self.item_component_and_content_emb_cols:
            meta_col, start_idx = meta[col]
            col_idx_init = np.arange(start_idx)
            inp_col = item_metadata[col].map(meta_col).values
            inp_col = np.concatenate([col_idx_init, inp_col])
            inp[col] = inp_col
        return inp
    
    def build_universe(self, item_metadata=None, combine=True):
        # self._set_item_component_cols()
        inp = self.build_universe_input(item_metadata)
        device = next(self.parameters()).device
        inp_pt = tree.tree_map(lambda x: torch.as_tensor(x, device=device)[None], inp)
        # out_universe = super(TabularSequenceFeatures, self).forward(inp_pt)
        out_universe = super().forward(inp_pt)
        out_universe = tree.tree_map(lambda x: x[0], out_universe)
    
        if combine:
            out_universe = torch.concat([out_universe[c] for c in self.item_component_and_content_out_cols], dim=1)
        return out_universe
    
    def update_items(self, item_metadata, ensure_no_new_components=False):
        """
        Helper that updates the input universe to be able to handle new items
        The input metadata should contain content ids and all required content embedding cols
        """
        
        if isinstance(item_metadata, cudf.DataFrame):
            item_metadata = item_metadata.to_pandas()
            
        # make sure the item_metadata contains all required columns
        for col in self.item_component_cols:
            if col not in item_metadata.columns:
                raise ValueError(f"item_metadata must contain all item_component_cat_cols, but {col} is missing")
        for col in self.content_emb_cols:
            if col not in item_metadata.columns:
                raise ValueError(f"item_metadata must contain all content_emb_cols, but {col} is missing")
        for col in self.item_component_and_content_out_cols:
            if col not in item_metadata.columns:
                raise ValueError(f"item_metadata must contain all item_component_and_content_out_cols, but {col} is missing")
            
        # extend the core item_id domain
        item_components_and_ids = self.schema.select_by_tag(tags=[CustomTags.ITEM_ID_COMPONENT, CustomTags.ITEM_COMPONENT]).column_names
        for col in item_components_and_ids:
            if col not in item_metadata.columns:
                raise ValueError(f"item_metadata must contain all item_component_cat_cols, but {col} is missing")
        item_metadata_components_and_ids = item_metadata[item_components_and_ids].drop_duplicates(subset=item_components_and_ids)
        extend_categorify_domain(self.schema, self.item_id, {col: item_metadata_components_and_ids[col].values for col in item_components_and_ids})
        self._cached_item_metadata = None
        
        # extend all required content ids
        for col in list(set(self.content_id_cols + self.content_emb_cols)):
            new_vals = item_metadata[col].drop_duplicates().values
            extend_categorify_domain(self.schema, col, new_vals)
            
        # get metas
        metas = self.get_item_cat_component_metadata(inverse=True)
        
        # make sure that all new components are present in the metas
        if ensure_no_new_components:
            for col in self.item_component_cols:
                meta_vals_new = item_metadata[col].unique()
                meta_vals_old = metas[col].index.values
                if not np.all(np.isin(meta_vals_new, meta_vals_old)):
                    raise ValueError(f"item_metadata must contain all item_component_cat_cols, but {col} is missing")
        
        # update pretrained embedding ops
        pemb = self.pretrained_embedding_ops_module
        lookup_key_to_indices_values = {}
        for col in self.content_emb_cols:
            out_col = pemb.col_map[col]
            item_metadata_unique = item_metadata[[col, out_col]].drop_duplicates(subset=[col])
            new_item_ids = item_metadata_unique[col]
            new_item_idxs = new_item_ids.map(metas[col])
            new_embs = np.stack(item_metadata_unique[out_col].values)
            lookup_key_to_indices_values[col] = (new_item_idxs, new_embs)
        pemb.update_embedding_tables(lookup_key_to_indices_values)
        
    def set_items_filter(self, item_filter):
        select_cols = self.schema.select_by_tag(
            [CustomTags.ITEM_ID_COMPONENT, CustomTags.ITEM_COMPONENT, CustomTags.CONTENT_ID]
        ).column_names
        self.item_filter = item_filter[[col for col in select_cols if col in item_filter.columns]]
        self._cached_item_metadata = None
        
    def clear_items_filter(self):
        """Clear the items filter and metadata cache. This will force a fresh load of the metadata on the next call to get_item_metadata."""
        self.item_filter = None
        self._cached_item_metadata = None
        
    def forward(self, inputs, training=False, testing=False, **kwargs):
        # self._set_item_component_cols()
        if self.item_id:
            self.item_seq = self.item_ids(inputs)
            
        # outputs = super(TabularSequenceFeatures, self).forward(inputs)
        outputs = super().forward(inputs)

        if self.masking or self.projection_module:
            outputs = self.aggregation(outputs)

        if self.projection_module:
            outputs = self.projection_module(outputs)

        if self.masking:
            outputs = self.masking(
                outputs,
                # item_ids=self.to_merge["categorical_module"].item_seq,
                item_ids=self.item_seq,
                training=training,
                testing=testing,
            )

        return outputs

    def project_continuous_features(self, dimensions):
        if isinstance(dimensions, int):
            dimensions = [dimensions]

        continuous = self.to_merge["continuous_module"]
        continuous.aggregation = "concat"

        continuous = SequentialBlock(
            continuous, MLPBlock(dimensions), AsTabular("continuous_projection")
        )

        self.to_merge["continuous_module"] = continuous

        return self

    def forward_output_size(self, input_size):
        output_sizes = {}
        for in_layer in self.merge_values:
            output_sizes.update(in_layer.forward_output_size(input_size))

        output_sizes = self._check_post_output_size(output_sizes)

        if self.projection_module:
            output_sizes = self.projection_module.output_size()

        return output_sizes


TabularFeaturesType = Union[TabularSequenceFeatures, TabularFeatures]
