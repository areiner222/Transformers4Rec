from typing import List, Optional, Union
from enum import Enum


class CustomTags(str, Enum):
    """
    Custom Tags that provide compatibility with Merlin's Tags enum.
    
    These tags are used for feature categorization and selection throughout the library.
    Each tag represents a specific type or semantic meaning of features.
    
    This enum includes additional custom tags beyond the standard Merlin tags.
    """    
    # Custom tags - add your own tags here
    CONTENT_ID = "CONTENT_ID"
    ITEM_ID_COMPONENT = "ITEM_ID_COMPONENT"
    ITEM_COMPONENT = "ITEM_COMPONENT"
    
