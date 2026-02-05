import io
from typing import Any, Optional

import torch
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint._nested_dict import (
    FLATTEN_MAPPING,
    flatten_state_dict,
)
from torch.distributed.checkpoint._sharded_tensor_utils import _flatten_sharded_tensors
from torch.distributed.checkpoint._traverse import set_element
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    STATE_DICT_TYPE,
    STORAGE_TYPES,
    StorageMeta,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    LoadPlanner,
    ReadItem,
)
from torch.distributed.checkpoint.planner_helpers import (
    _create_read_items,
    _init_state_dict,
)
from torch.distributed.checkpoint.utils import find_state_dict_object
from torch.distributed.tensor import DTensor

class DistCPDefaultLoadPlanner(LoadPlanner):
    """
    DefaultLoadPlanner that adds multiple features on top of LoadPlanner.

    In particular it adds the following:

    flatten_state_dict: Handle state_dict with nested dicts
    flatten_sharded_tensors: For FSDP in 2D parallel mode
    allow_partial_load: If False, will raise a runtime error if a key is present in state_dict, but not in the checkpoint.
    """

    original_state_dict: STATE_DICT_TYPE
    mappings: FLATTEN_MAPPING

    def __init__(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        allow_partial_load: bool = False,
    ) -> None:
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors
        self.original_state_dict = {}
        self.mappings = {}
        self.allow_partial_load = allow_partial_load

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Optional[Metadata] = None,
        is_coordinator: bool = False,
    ) -> None:
        _init_state_dict(state_dict)
        self.original_state_dict = state_dict

        if self.flatten_sharded_tensors:
            state_dict = _flatten_sharded_tensors(state_dict)

        if self.flatten_state_dict:
            state_dict, self.mappings = flatten_state_dict(state_dict)

        self.state_dict = state_dict
        self.metadata = metadata
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> LoadPlan:
        assert self.metadata is not None
        if self.flatten_state_dict:
            current_keys = set(self.state_dict.keys())
            load_keys = set(self.metadata.state_dict_metadata.keys())
            missing_keys = load_keys - current_keys
            if missing_keys:
                old_state_dict, old_mappings = flatten_state_dict(
                    self.original_state_dict
                )
                old_keys = set(old_state_dict.keys())
                if old_keys & missing_keys:
                    self.state_dict, self.mappings = old_state_dict, old_mappings

        return create_default_local_load_plan(
            self.state_dict, self.metadata, not self.allow_partial_load
        )

    def create_global_plan(self, global_plan: list[LoadPlan]) -> list[LoadPlan]:
        return create_default_global_load_plan(global_plan)

    def finish_plan(self, new_plan: LoadPlan) -> LoadPlan:
        return new_plan

    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        if self.flatten_state_dict:
            set_element(
                self.original_state_dict,
                self.mappings[read_item.dest_index.fqn],
                torch.load(value, weights_only=False),
            )
        else:
            self.state_dict[read_item.dest_index.fqn] = torch.load(
                value, weights_only=False
            )

    def resolve_tensor(self, read_item: ReadItem):
        tensor = self.lookup_tensor(read_item.dest_index)
        return self.transform_tensor(read_item, tensor)

    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        pass

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        """Extension from the planner interface to make it easy to extend the default planner."""
        return find_state_dict_object(self.state_dict, index)

    def transform_tensor(self, read_item: ReadItem, tensor: torch.Tensor):
        """Extension from the planner interface to make it easy to extend the default planner."""
        return narrow_tensor_by_index(tensor, read_item.dest_offsets, read_item.lengths)

def create_default_local_load_plan(
    state_dict: dict[str, Any], metadata: Metadata, strict: bool = True
) -> LoadPlan:
    requests = []
    """
    Create the ``LoadPlan`` used by DefaultLoadPlanner.

    It produces one read item per value in ``state_dict`` using the metadata in ``metadata``.

    The default behavior is to match key exactly between state_dict and metadata.
    It handles resharding by issuing multiple read requests against storage in order to match
    load requirements.
    """
    current_keys = set(state_dict.keys())
    load_keys = set(metadata.state_dict_metadata.keys())
    unexpected_keys = load_keys - current_keys
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys in checkpoint state_dict: {unexpected_keys}.")

    for fqn, obj in state_dict.items():
        # ignore state_dict keys which do not exist in `state_dict` if strict=False
        if fqn not in metadata.state_dict_metadata:
            if strict:
                raise RuntimeError(f"Missing key in checkpoint state_dict: {fqn}.")
            else:
                continue

        md = metadata.state_dict_metadata[fqn]
        if (
            isinstance(md, TensorStorageMetadata)
            and getattr(obj, "size", None) is not None
            and md.size != obj.size()
        ):
            raise ValueError(
                f"Size mismatch between saved {md.size} and current: {obj.size()} for {fqn}",
            )
        # Since DTensor supports submesh, adding extra check to ensure _create_read_items()
        # gets called only when the current rank is part of the mesh for the corresponding DTensor.
        if isinstance(obj, DTensor):
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_read_items(fqn, md, obj)
        else:
            requests += _create_read_items(fqn, md, obj)

    return LoadPlan(requests)


def create_default_global_load_plan(
    all_plans: list[LoadPlan],
) -> list[LoadPlan]:
    """
    Create global load plan used by DefaultLoadPlanner.

    The default load behavior involved no global coordination and this function
    currently doesn't change the local plans.
    """
    return all_plans

