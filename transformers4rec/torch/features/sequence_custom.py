from typing import Dict, List, Optional, Tuple, Union

import torch
from merlin.models.utils.doc_utils import docstring_parameter
from merlin.schema import Tags, TagsType

from merlin_standard_lib import Schema
from merlin.dataloader.ops.embeddings import EmbeddingOperator

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

from .tabular_custom import TabularFeaturesCustom
from .sequence import SequenceEmbeddingFeatures


class TabularSequenceFeatures(TabularFeaturesCustom):
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
        self.projection_module = projection_module
        self.set_masking(masking)

    @classmethod
    def from_schema(  # type: ignore
        cls,
        schema: Schema,
        continuous_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.CONTINUOUS,),
        categorical_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.CATEGORICAL,),
        pretrained_embeddings_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.EMBEDDING,),
        pretrained_embedding_ops: Optional[List[EmbeddingOperator]] = None,
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
        output: TabularSequenceFeatures = super().from_schema(  # type: ignore
            schema=schema,
            continuous_tags=continuous_tags,
            categorical_tags=categorical_tags,
            pretrained_embeddings_tags=pretrained_embeddings_tags,
            pretrained_embedding_ops=pretrained_embedding_ops,
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
        if masking and not getattr(output, "item_id", None):
            raise ValueError("For masking a categorical_module is required including an item_id.")
        output.set_masking(masking)  # type: ignore

        return output

    @property
    def masking(self):
        return self._masking

    def set_masking(self, value):
        self._masking = value

    @property
    def item_id(self) -> Optional[str]:
        if "categorical_module" in self.to_merge:
            return getattr(self.to_merge["categorical_module"], "item_id", None)

        return None

    @property
    def item_embedding_table(self) -> Optional[torch.nn.Module]:
        if "categorical_module" in self.to_merge:
            return getattr(self.to_merge["categorical_module"], "item_embedding_table", None)

        return None

    def forward(self, inputs, training=False, testing=False, **kwargs):
        outputs = super(TabularSequenceFeatures, self).forward(inputs)

        if self.masking or self.projection_module:
            outputs = self.aggregation(outputs)

        if self.projection_module:
            outputs = self.projection_module(outputs)

        if self.masking:
            outputs = self.masking(
                outputs,
                item_ids=self.to_merge["categorical_module"].item_seq,
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

