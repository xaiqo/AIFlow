# Plugin System

Goals
- Allow third parties to add passes, backends, profilers, and visualizations without modifying core.

Mechanism
- Lightweight registry: components register by name and type.
- Discovery: entry points or explicit Python import in user config.

Guidelines
- Keep plugin dependencies isolated.
- Provide clear docs and versioned compatibility constraints.




