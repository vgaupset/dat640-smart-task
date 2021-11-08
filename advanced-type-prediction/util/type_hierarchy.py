#!/usr/bin/env python
# coding: utf-8

# In[3]:


import math
import operator
from typing import Callable, List, Optional, Set


# In[4]:


class TypeTaxonomy:
    
    ROOT = "owl:Thing"
    
    def __init__(self, tsv_filename: str) -> None:
        """Initializes the type taxonomy by loading it from a TSV file.
        
        Args:
            tsv_filename: Name of TSV file, with type_id, depth, and parent_id columns.
        """
        self._types = {self.ROOT: {"parent": None, "depth": 0}}
        self._max_depth = 0
        with open(tsv_filename, "r") as tsv_file:
            next(tsv_file)  # Skip header row
            for line in tsv_file:
                fields = line.rstrip().split("\t")
                type_id, depth, parent_type = fields[0], int(fields[1]), fields[2]
                self._types[type_id] = {"parent": parent_type, "depth": depth}
                self._max_depth = max(depth, self._max_depth)
                
        # Once all types have been read in, we also populate each type with a list
        # of its children for convenience (if the taxonomy is to be traversed
        # downwards not just upwards).
        for type_id in self._types:
            if type_id == self.ROOT:
                continue
            parent_type = self._types[type_id]["parent"]            
            if "children" not in self._types[parent_type]:
                self._types[parent_type]["children"] = set()
            self._types[parent_type]["children"].add(type_id)
                        
    def max_depth(self) -> int:
        """Returns the maximum depth of the type taxonomy."""
        return self._max_depth
    
    def is_root(self, type_id: str) -> bool:
        """Returns true if the type is the root of the taxonomy.
        
        Args:
            type_id: Type ID.
            
        Returns:
            True if root.
        """
        return type_id == self.ROOT
    
    def depth(self, type_id: str) -> int:
        """Returns the depth of a type in the taxonomy.
        
        Args:
            type_id: Type ID.
            
        Returns:
            The depth of the type in the hierarchy (0 for root).
        """
        return self._types.get(type_id, {}).get("depth")

    def parent(self, type_id: str) -> Optional[str]:
        """Returns the parent type of a type in the taxonomy.
        
        Args:
            type_id: Type ID.
            
        Returns:
            Parent type ID, or None if the input type is root.
        """
        return self._types.get(type_id, {}).get("parent")

    def children(self, type_id: str) -> Set[str]:
        """Returns the set of children types of a type in the taxonomy.
        
        Args:
            type_id: Type ID.
            
        Returns:
            Set of type IDs (empty set if leaf type).
        """
        return self._types.get(type_id, {}).get("children", set())
    
    def dist(self, type_id1: str, type_id2: str) -> float:
        """Computes the distance between two types in the taxonomy.
        
        Args:
            type_id1: ID of first type.
            type_id2: ID of second type.
            
        Returns:
            The distance between the two types in the type taxonomy, which is
            the number of steps between them if they lie on the same branch,
            and otherwise `math.inf`.
        """
        # Find which type has higher depth and set if to type_a; the other is type_b.        
        type_a, type_b = (type_id2, type_id1) if self.depth(type_id1) < self.depth(type_id2)                          else (type_id1, type_id2)
        dist = self.depth(type_a) - self.depth(type_b)
        
        # If they lie on the same branch, then when traversing type_a for dist steps
        # would make us end up with type_b; otherwise, they're not on the same branch.
        for _ in range(dist):
            type_a = self.parent(type_a)
        
        return dist if type_a == type_b else math.inf


# In[5]:


typeobj=TypeTaxonomy("../data/dbpedia_types.tsv")


# In[6]:


typeobj._types


# In[ ]:




