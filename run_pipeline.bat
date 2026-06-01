@echo off
cd /d "E:\Naad\Projects\Real-Time-Social-Media-Monitoring-of-Rupiah-Exchange-Rate-Issues-Using-Transformer-Based-NLP"
call venv\Scripts\activate.bat
cd src
python pipeline.py >> ..\data\pipeline.log 2>&1
