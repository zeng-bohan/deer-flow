from deerflow.storage.backends.local_fs.local_fs_store import LocalFsBlobStore

# Sentinel the folder-scan factory looks for (mirrors memory backends' MANAGER_CLASS).
STORE_CLASS = LocalFsBlobStore

__all__ = ["LocalFsBlobStore", "STORE_CLASS"]
