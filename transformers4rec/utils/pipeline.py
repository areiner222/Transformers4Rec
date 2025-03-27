# top
import optree as tree
import numpy as np
import pandas as pd
import cupy as cp
import cudf
import nvtabular as nvt
from ..config.tags import CustomTags
from nvtabular.ops import TagAsItemID, TagAsItemFeatures

from merlin.core.dispatch import get_lib
from merlin.dataloader.ops.embeddings import EmbeddingOperator
pd_lib = get_lib()
# np_lib = cp if pd_lib == cudf else np


def prepare_item_cols(content_id_cols, item_component_cols, other_item_id_cols=None, item_id_out_col='_item_id', verbose=False):
    if verbose:
        print(f"Preparing item columns with content_id_cols={content_id_cols}, item_component_cols={item_component_cols}")
    
    # ensure list
    other_item_id_cols = tree.tree_leaves(other_item_id_cols)
    content_id_cols = tree.tree_leaves(content_id_cols)
    item_id_cols = list(set(content_id_cols + other_item_id_cols))
    item_component_cols = list(set(tree.tree_leaves(item_component_cols)))
    
    if verbose:
        print(f"Processed column lists: item_id_cols={item_id_cols}, item_component_cols={item_component_cols}")
    
    # check that no component col is a contnet col
    for col in item_component_cols:
        if col in content_id_cols:
            raise ValueError(f"component col {col} cannot be a content identifying col")

    # encode unique identifier with component cols
    if verbose:
        print(f"Creating unique item identifier from columns: {item_id_cols + [c for c in item_component_cols if c not in item_id_cols]}")
    
    item = (
        [tuple(item_id_cols + [c for c in item_component_cols if c not in item_id_cols])]
        >> nvt.ops.Categorify(encode_type='combo') 
        >> nvt.ops.Rename(name=item_id_out_col) >> TagAsItemID()
    )
    
    # we need to make sure all item_id_cols that are not components are also encoded uniquely
    if content_id_cols:
        if verbose:
            print(f"Encoding content ID columns: {content_id_cols}")
        content_ids = content_id_cols >> nvt.ops.Categorify() >> nvt.ops.AddTags(tags=[CustomTags.ITEM_ID_COMPONENT, CustomTags.CONTENT_ID])
    else:
        content_ids = []
        if verbose:
            print("No content ID columns provided")
    
    other_item_id_cols_not_components = [col for col in item_id_cols if col not in item_component_cols and col not in content_id_cols]
    if other_item_id_cols_not_components:
        if verbose:
            print(f"Encoding other item ID columns: {other_item_id_cols_not_components}")
        other_item_ids = other_item_id_cols_not_components >> nvt.ops.Categorify() >> nvt.ops.AddTags(CustomTags.ITEM_ID_COMPONENT)
    else:
        other_item_ids = []
        if verbose:
            print("No other item ID columns that are not components")
    
    item = item + content_ids + other_item_ids
    
    # build item embeddings that are components
    if verbose:
        print(f"Building item component embeddings for columns: {item_component_cols}")
    item_emb = item_component_cols >> nvt.ops.Categorify() >> TagAsItemFeatures() >> nvt.ops.AddTags(CustomTags.ITEM_COMPONENT)
    
    # tag all item embeddings that are id cols with the item id tag
    other_item_id_cols_are_components = [col for col in item_id_cols if col in item_component_cols and col not in content_id_cols]
    _rem_cols = [col for col in item_component_cols if col not in other_item_id_cols_are_components]
    item_emb = (item_emb[other_item_id_cols_are_components] >> nvt.ops.AddTags(CustomTags.ITEM_ID_COMPONENT)) + item_emb[_rem_cols]
    
    if verbose:
        print("Item columns preparation completed")
    
    return item, item_emb


def build_embedding_op(content_id_col, df_item_content, workflow, content_col='content'):    
    # get the item categorical info
    item_meta = pd_lib.read_parquet(workflow.output_schema[content_id_col].properties['cat_path'])

    # merge
    item_to_vec = pd_lib.merge(item_meta.reset_index(), df_item_content, on=content_id_col).set_index('index').sort_index()
    item_cardinality = workflow.output_schema[content_id_col].int_domain.max + 1

    idx = item_to_vec.index.values.get()
    content_vecs_s = item_to_vec[content_col]
    if not isinstance(content_vecs_s, pd.Series):
        content_vecs_s = content_vecs_s.to_pandas()
    content_vecs = np.stack(content_vecs_s).astype('float32')
    embeddings = np.zeros((item_cardinality, content_vecs.shape[-1])).astype('float32')
    embeddings[idx] = content_vecs
    return EmbeddingOperator(embeddings, lookup_key=content_id_col, embedding_name=content_col)


def extend_categorify_domain(schema, column_name, new_values):
    """
    Extend the domain of a Categorify node for a specific column at inference time.
    Supports both single columns and combo (multi-column) categorification.
    
    Parameters:
    -----------
    schema : nvt.Schema
        The schema containing the column
    column_name : str
        Name of the column to extend in the schema
    new_values : list or dict
        For single columns: list of values to add to the mapping
        For combo columns: dictionary mapping column names to arrays/lists of values
        Example: {'product_id': ['id1', 'id2'], 'color': ['red', 'blue']}
    
    Returns:
    --------
    Updated mapping dataframe
    """
    # get the cat path
    cat_path = schema[column_name].properties['cat_path']

    # Read the existing mapping
    mapping_df = pd.read_parquet(cat_path)
    
    # Determine if this is a combo column based on new_values structure
    is_combo = isinstance(new_values, dict)
    
    if is_combo:
        # Get the component columns from the dictionary keys
        component_cols = list(new_values.keys())
        
        # Convert new_values to DataFrame - it's already in the right format
        new_items_df = pd.DataFrame(new_values)
        
        # Check which combinations are new
        existing_combos = set()
        for _, row in mapping_df[component_cols].iterrows():
            # Convert numpy values to Python types for hashing
            combo = tuple(float(v) if isinstance(v, (np.floating, np.integer)) else v for v in row[component_cols])
            existing_combos.add(combo)
        
        # Filter to only new combinations
        new_combos = []
        for _, row in new_items_df.iterrows():
            # Convert numpy values to Python types for hashing
            combo = tuple(float(v) if isinstance(v, (np.floating, np.integer)) else v for v in row[component_cols])
            if combo not in existing_combos and not any(pd.isna(v) for v in combo):
                new_combos.append({col: row[col] for col in component_cols})
        
        if not new_combos:
            return mapping_df  # No new items to add
        
        # Create DataFrame for new mappings
        new_mappings = pd.DataFrame(new_combos)
    else:
        # Single column case
        # Get new values not in the mapping
        existing_values = set(mapping_df[column_name].values)
        new_items = [v for v in new_values if v not in existing_values and not pd.isna(v)]
        
        if not new_items:
            return mapping_df  # No new items to add
        
        # Create DataFrame for new mappings
        new_mappings = pd.DataFrame({column_name: new_items})
    
    # Get the next available index
    max_index = mapping_df.index.max() if not mapping_df.empty else -1
    
    # Assign indices starting from max_index + 1
    new_mappings.index = range(max_index + 1, max_index + 1 + len(new_mappings))
    
    # Concatenate with existing mapping
    updated_mapping = pd.concat([mapping_df, new_mappings])
    
    # Save updated mapping
    updated_mapping.to_parquet(cat_path)
    
    return updated_mapping


def get_category_op(workflow, col):
    # Find the Categorify operator in the workflow
    categorify_op = None
    for node in workflow.graph.get_nodes_by_op_type([workflow.output_node], nvt.ops.Categorify):
        if col in node.output_columns.names:
            return categorify_op
    return categorify_op


# def prepare_item_cols(item_id_cols, item_component_cols, content_id_cols=None, item_id_out_col='_item_id'):
#     # ensure list
#     item_id_cols = tree.tree_leaves(item_id_cols)
#     item_component_cols = tree.tree_leaves(item_component_cols)

#     # encode unique identifier with component cols
#     item_ids = item_id_cols >> nvt.ops.AddTags(CustomTags.ITEM_ID_COMPONENT)
#     item = (
#         [tuple(item_id_cols + item_component_cols)]
#         >> nvt.ops.Categorify(encode_type='combo') 
#         >> nvt.ops.Rename(name=item_id_out_col) >> TagAsItemID()
#     )
#     item = item + item_ids
    
#     # build item embeddings that are components
#     item_emb = item_component_cols >> nvt.ops.Categorify() >> TagAsItemFeatures() >> nvt.ops.AddTags(CustomTags.ITEM_COMPONENT)

#     # we also should categorify all item_id_cols that aren't explicitly content id cols
#     # if no content_id_cols provided, we assume all item_id_cols are reuquired to uniquely identify contnet
#     if content_id_cols is not None:
#         content_id_cols = tree.tree_leaves(content_id_cols)
#         for col in content_id_cols:
#             if col not in item_id_cols:
#                 raise ValueError(f"content_id_cols must be a subset of item_id_cols, but {col} is not in {item_id_cols}")
#         item_ids_to_embed = [col for col in item_id_cols if col not in content_id_cols]
#         item_ids_emb = item_ids_to_embed >> nvt.ops.Categorify() >> TagAsItemFeatures() >> nvt.ops.AddTags(CustomTags.ITEM_COMPONENT)
#         item_emb += item_ids_emb
    
#     return item, item_emb
    
    
# def extend_categorify_domain(schema, column_name, new_values):
#     """
#     Extend the domain of a Categorify node for a specific column at inference time.
    
#     Parameters:
#     -----------
#     workflow : nvt.Workflow
#         The workflow containing the Categorify op
#     column_name : str
#         Name of the column to extend
#     new_values : list or pandas.Series
#         New values to add to the mapping
    
#     Returns:
#     --------
#     Updated workflow with extended categorical domain
#     """
#     # get the cat path
#     cat_path = schema[column_name].properties['cat_path']

#     # Read the existing mapping
#     mapping_df = pd.read_parquet(cat_path)
    
#     # Get new values not in the mapping
#     existing_values = set(mapping_df[column_name].values)
#     new_items = [v for v in new_values if v not in existing_values and not pd.isna(v)]
    
#     if not new_items:
#         return mapping_df  # No new items to add
    
#     # Get the next available index
#     max_index = mapping_df.index.max()
    
#     # Create DataFrame for new mappings
#     new_mappings = pd.DataFrame({column_name: new_items})
    
#     # Assign indices starting from max_index + 1
#     new_mappings.index = range(max_index + 1, max_index + 1 + len(new_items))
    
#     # Concatenate with existing mapping
#     updated_mapping = pd.concat([mapping_df, new_mappings])
    
#     # Save updated mapping
#     updated_mapping.to_parquet(cat_path)
    
#     return updated_mapping