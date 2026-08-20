"""Vendored TAM inference code (pruned public TAM release, imports rewritten).

Only the surfaces needed for deployment on robot-control-stack are included:
checkpoint loading and adaptor export (``deploy.inf_util``), the streaming
history encoder (``deploy.history_runtime``), and their model/physics
dependencies. Training, data generation, and transport code are not part of
this package.
"""
