# PROD-1 Kill-Switch Design

The kill-switch is enabled by default, can be tripped, blocks shadow activation when tripped, treats disabled/unknown states as invalid or fail-closed, and requires dry-run approval metadata before reset. Reset does not activate runtime.
