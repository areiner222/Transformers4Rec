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


from merlin.schema import Tags

import transformers4rec.torch as tr

from ...conftest import tabular_schemas


@tabular_schemas()
def test_tabular_features(schema, torch_tabular_data):
    tab_module = tr.TabularFeatures.from_schema(schema)

    outputs = tab_module(torch_tabular_data)

    assert set(outputs.keys()) == set(
        schema.select_by_tag(Tags.CONTINUOUS).column_names
        + schema.select_by_tag(Tags.CATEGORICAL).column_names
    )


@tabular_schemas()
def test_tabular_features_embeddings_options(schema, torch_tabular_data):
    EMB_DIM = 100
    tab_module = tr.TabularFeatures.from_schema(schema, embedding_dim_default=EMB_DIM)

    outputs = tab_module(torch_tabular_data)

    categ_features = schema.select_by_tag(Tags.CATEGORICAL).column_names
    assert all(v.shape[-1] == EMB_DIM for k, v in outputs.items() if k in categ_features)


@tabular_schemas()
def test_tabular_features_with_projection(schema, torch_tabular_data):
    tab_module = tr.TabularFeatures.from_schema(schema, continuous_projection=64)

    outputs = tab_module(torch_tabular_data)

    continuous_feature_names = schema.select_by_tag(Tags.CONTINUOUS).column_names

    assert len(set(continuous_feature_names).intersection(set(outputs.keys()))) == 0
    assert "continuous_projection" in outputs
    assert list(outputs["continuous_projection"].shape)[1] == 64


@tabular_schemas()
def test_tabular_features_soft_encoding(schema, torch_tabular_data):
    emb_cardinality = 10
    emb_dim = 8
    tab_module = tr.TabularFeatures.from_schema(
        schema,
        continuous_soft_embeddings=True,
        soft_embedding_cardinality_default=emb_cardinality,
        soft_embedding_dim_default=emb_dim,
    )

    outputs = tab_module(torch_tabular_data)

    assert (
        list(outputs.keys())
        == schema.select_by_tag(Tags.CONTINUOUS).column_names
        + schema.select_by_tag(Tags.CATEGORICAL).column_names
    )

    assert all(
        list(outputs[col_name].shape) == list(torch_tabular_data[col_name].shape) + [emb_dim]
        for col_name in schema.select_by_tag(Tags.CONTINUOUS).column_names
    )
