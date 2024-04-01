#!/bin/bash


exec .venv/bin/python -m streamlit run streamlit_run.py --server.port ${API_PORT}
