"""Rerun recordings (.rrd, one per episode) to the LeRobot v2.1 layout Dream's hackathon kit records.

    uv run --group convert convert inspect <file.rrd>                    # what is in it, and a draft mapping
    uv run --group convert convert run <files.rrd...> <out> --map <map.yaml>

The mapping file (configs/convert/*.yaml) says which recorded entity and scalar names fill each part of the state and
action vectors, in which unit, and which entity is each camera. Nothing about the recorder is in the code.
"""
