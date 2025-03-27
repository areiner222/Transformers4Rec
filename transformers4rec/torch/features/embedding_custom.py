# from
from typing import List, Optional, Union, Dict

import torch
from merlin.dataloader.ops.embeddings import EmbeddingOperator
from merlin.schema import Tags, TagsType
from merlin_standard_lib import Schema, categorical_cardinalities
from ..block.base import SequentialBlock
from ..tabular.base import (
    TABULAR_MODULE_PARAMS_DOCSTRING,
    FilterFeatures,
    TabularAggregationType,
    TabularTransformation,
    TabularTransformationType,
)
from .base import InputBlock


class PretrainedEmbeddingFeaturesCustom(InputBlock):
    def __init__(
        self, 
        embedding_ops: List[EmbeddingOperator], 
        pretrained_output_dims: Optional[Union[int, Dict[str, int]]] = None, 
        sequence_combiner: Optional[Union[str, torch.nn.Module]] = None, 
        pre: Optional[TabularTransformationType] = None, 
        post: Optional[TabularTransformationType] = None, 
        aggregation: Optional[TabularAggregationType] = None, 
        normalizer: Optional[TabularTransformationType] = None, 
        schema: Optional[Schema] = None, 
        **kwargs
    ):
        if isinstance(normalizer, str):
            normalizer = TabularTransformation.parse(normalizer)
        if not post:
            post = normalizer
        elif normalizer:
            post = SequentialBlock(normalizer, post)  # type: ignore

        super().__init__(pre=pre, post=post, aggregation=aggregation, schema=schema)
        self.embedding_ops = embedding_ops
        self.input_names = [op.lookup_key for op in self.embedding_ops]
        self.features = [op.embedding_name for op in self.embedding_ops]
        self.filter_features = FilterFeatures(self.input_names)
        self.pretrained_output_dims = pretrained_output_dims
        self.sequence_combiner = self.parse_combiner(sequence_combiner)
        self.col_map = {op.lookup_key: op.embedding_name for op in self.embedding_ops}
        self.inp_to_dim = {op.lookup_key: op.embeddings.shape[-1] for op in self.embedding_ops}
        
        embedding_tables = {}
        for op in self.embedding_ops:
            if op.embedding_name not in embedding_tables:
                embedding_tables[op.lookup_key] = self.embedding_op_to_embedding_module(op)

        self.embedding_tables = torch.nn.ModuleDict(embedding_tables)
        
    def embedding_op_to_embedding_module(self, embedding_op: EmbeddingOperator) -> torch.nn.Embedding:
        return torch.nn.Embedding(*embedding_op.embeddings.shape, _weight=torch.as_tensor(embedding_op.embeddings))

    def build(self, input_size, **kwargs):
        if input_size is not None:
            if self.pretrained_output_dims:
                self.projection = torch.nn.ModuleDict()
                for key_inp, key_out in self.col_map.items():
                    pretrained_output_dim = (
                        self.pretrained_output_dims[key_inp] if isinstance(self.pretrained_output_dims, dict) 
                        else self.pretrained_output_dims
                    )
                    self.projection[key_out] = torch.nn.Linear(
                        input_size[key_inp][-1], pretrained_output_dim
                    )
                
        return super().build(input_size, **kwargs)

    # @classmethod
    # def from_schema(
    #     cls,
    #     schema: Schema,
    #     tags: Optional[TagsType] = None,
    #     pretrained_output_dims=None,
    #     sequence_combiner=None,
    #     normalizer: Optional[Union[str, TabularTransformationType]] = None,
    #     pre: Optional[TabularTransformationType] = None,
    #     post: Optional[TabularTransformationType] = None,
    #     aggregation: Optional[TabularAggregationType] = None,
    #     **kwargs,
    # ):  # type: ignore
    #     if tags:
    #         schema = schema.select_by_tag(tags)

    #     features = schema.column_names
    #     return cls(
    #         features=features,
    #         pretrained_output_dims=pretrained_output_dims,
    #         sequence_combiner=sequence_combiner,
    #         pre=pre,
    #         post=post,
    #         aggregation=aggregation,
    #         normalizer=normalizer,
    #     )

    def forward(self, inputs):
        # get the inputs
        filtered_inputs = self.filter_features(inputs)
        
        # compute the output embeddings
        output = {}
        for name, val in filtered_inputs.items():
            out_name = self.col_map[name]
            if isinstance(val, tuple):
                values, offsets = val
                values = torch.squeeze(values, -1)
                # for the case where only one value in values
                if len(values.shape) == 0:
                    values = values.unsqueeze(0)
                output[out_name] = self.embedding_tables[name](values, offsets[:, 0])
            else:
                # if len(val.shape) <= 1:
                #    val = val.unsqueeze(0)
                output[out_name] = self.embedding_tables[name](val)
        
        if self.pretrained_output_dims:
            output = {key: self.projection[key](val) for key, val in output.items()}
        
        if self.sequence_combiner:
            for key, val in output.items():
                if val.dim() > 2:
                    output[key] = self.sequence_combiner(val, axis=1)

        return output

    # def forward_output_size(self, input_sizes):
    #     sizes = self.filter_features.forward_output_size(input_sizes)
    #     if self.pretrained_output_dims:
    #         if isinstance(self.pretrained_output_dims, dict):
    #             sizes.update(
    #                 {
    #                     key: torch.Size(list(sizes[key][:-1]) + [self.pretrained_output_dims[key]])
    #                     for key in self.features
    #                 }
    #             )
    #         else:
    #             sizes.update(
    #                 {
    #                     key: torch.Size(list(sizes[key][:-1]) + [self.pretrained_output_dims])
    #                     for key in self.features
    #                 }
    #             )
    #     return sizes
    
    def forward_output_size(self, input_sizes):
        sizes = {}

        for fname, emb_name in self.col_map.items():
            fshape = input_sizes[fname]
            sizes[emb_name] = torch.Size(list(fshape) + [self.inp_to_dim[fname]])

        return sizes

    def parse_combiner(self, combiner):
        if isinstance(combiner, str):
            if combiner == "sum":
                combiner = torch.sum
            elif combiner == "max":
                combiner = torch.max
            elif combiner == "min":
                combiner = torch.min
            elif combiner == "mean":
                combiner = torch.mean
        return combiner
