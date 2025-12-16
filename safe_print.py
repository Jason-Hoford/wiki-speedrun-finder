"""
Safe printing utility for handling Unicode characters on Windows terminals.
Prevents UnicodeEncodeError when printing non-ASCII characters.
"""
import sys


def safe_print(*args, **kwargs):
    """
    Print function that handles Unicode encoding errors gracefully.
    Replaces characters that can't be encoded with '?' instead of crashing.
    """
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Convert all arguments to strings and encode safely
        safe_args = []
        for arg in args:
            try:
                # Try to encode and decode to catch issues
                str_arg = str(arg)
                # Replace characters that can't be encoded
                safe_str = str_arg.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
                safe_args.append(safe_str)
            except Exception:
                safe_args.append('[unprintable]')
        print(*safe_args, **kwargs)


def configure_console_encoding():
    """
    Try to configure the console to use UTF-8 encoding.
    This may not work on all systems, so we catch exceptions.
    """
    try:
        if sys.platform == 'win32':
            # Try to set UTF-8 encoding on Windows
            import os
            os.system('chcp 65001 > nul 2>&1')
    except Exception:
        pass  # Silently fail if we can't change encoding
