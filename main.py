#!/usr/bin/env python3
"""Entry point for the OGrE work-intelligence simulation.

    python main.py                 # default model (llama3.2:3b) / heuristic fallback
    python main.py --model qwen3:8b
    OGRE_MODEL=llama3.2:3b python main.py
"""

from ogre.sim import main

if __name__ == "__main__":
    main()
