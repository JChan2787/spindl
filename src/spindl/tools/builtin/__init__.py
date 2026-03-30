"""
Built-in Tools Package.

This package contains tools that ship with spindl.
Each tool is in its own subpackage with an __init__.py that
exports a Tool subclass.

Structure:
    builtin/
    ├── __init__.py          (this file)
    ├── screen_vision/
    │   └── __init__.py      (exports ScreenVisionTool)
    └── future_tool/
        └── __init__.py      (exports FutureTool)

Adding a new built-in tool:
    1. Create a new directory under builtin/
    2. Add an __init__.py that exports a class inheriting from Tool
    3. The directory name becomes the tool name in config
"""
