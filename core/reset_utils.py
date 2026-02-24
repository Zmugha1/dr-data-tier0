"""
Emergency reset utilities - completely isolated from ChromaDB
"""
import os
import shutil
import gc
import time
from pathlib import Path


def force_delete_windows(path):
    """Force delete on Windows with multiple strategies"""
    path = Path(path)
    if not path.exists():
        return True

    # Strategy 1: Normal delete with retries
    for i in range(5):
        try:
            shutil.rmtree(path, ignore_errors=False)
            return True
        except PermissionError:
            gc.collect()
            time.sleep(0.5)
        except Exception:
            break

    # Strategy 2: Rename then delete (breaks locks)
    try:
        temp_name = path.parent / f"{path.name}_old_{int(time.time())}"
        path.rename(temp_name)
        shutil.rmtree(temp_name, ignore_errors=True)
        return True
    except Exception:
        pass

    # Strategy 3: Individual file deletion
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            for file in files:
                try:
                    os.unlink(os.path.join(root, file))
                except Exception:
                    pass
            for dir_name in dirs:
                try:
                    os.rmdir(os.path.join(root, dir_name))
                except Exception:
                    pass
        os.rmdir(str(path))
        return True
    except Exception:
        pass

    return False


def reset_vector_db_nuclear():
    """
    Nuclear option: Delete Vector DB without importing ANYTHING from the app
    Returns: (success: bool, message: str)
    """
    db_path = Path("data/vector_db")

    if not db_path.exists():
        return True, "No DB to reset"

    # Close any potential file handles via GC
    gc.collect()
    time.sleep(0.3)

    # Attempt deletion
    if force_delete_windows(db_path):
        # Recreate empty
        db_path.mkdir(parents=True, exist_ok=True)
        return True, "Reset successful"
    else:
        return False, "Could not delete - files locked by OS"


def reset_all_data_nuclear():
    """
    Nuclear full reset: Vector DB + Knowledge Graph.
    Returns: (success: bool, message: str)
    """
    db_path = Path("data/vector_db")
    kg_path = Path("data/graph_store")

    gc.collect()
    time.sleep(0.3)

    vec_ok = True
    if db_path.exists():
        vec_ok = force_delete_windows(db_path)
        if vec_ok:
            db_path.mkdir(parents=True, exist_ok=True)

    if kg_path.exists():
        force_delete_windows(kg_path)
        kg_path.mkdir(parents=True, exist_ok=True)

    if vec_ok:
        return True, "Full reset successful"
    return False, "Could not delete vector DB - files locked by OS"


def verify_reset():
    """Verify the DB path is clean"""
    db_path = Path("data/vector_db")
    if not db_path.exists():
        return True
    # Check if it's truly empty or just has empty folders
    contents = list(db_path.iterdir())
    return len(contents) == 0
