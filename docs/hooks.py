"""MkDocs hooks for SMLR documentation.

With notebook execution enabled (mkdocs-jupyter execute: true), the complex
metric generation is no longer needed. Notebooks are the source of truth.
"""


def on_pre_build(config):
    """Pre-build hook - minimal setup."""
    print("\n" + "=" * 60)
    print("Building SMLR documentation")
    print("Notebooks will be executed by mkdocs-jupyter")
    print("=" * 60)


def on_env(env, config, files):
    """Add custom Jinja filters for formatting."""
    
    def format_percent(value, decimals=1):
        """Format a decimal as percentage."""
        try:
            return f"{float(value) * 100:.{decimals}f}%"
        except (ValueError, TypeError):
            return str(value)
    
    def format_decimal(value, decimals=2):
        """Format a decimal number."""
        try:
            return f"{float(value):.{decimals}f}"
        except (ValueError, TypeError):
            return str(value)
    
    env.filters['percent'] = format_percent
    env.filters['decimal'] = format_decimal
    
    return env
