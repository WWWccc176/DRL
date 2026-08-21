from __future__ import annotations

import os

# Must be set before importing scripts.agent.a11 so config.py is initialized
# with the full profile in the main process and all spawned workers.
os.environ["A11_TRAIN_PROFILE"] = "full"

from scripts.agent.a11.main import main


if __name__ == "__main__":
    main()
