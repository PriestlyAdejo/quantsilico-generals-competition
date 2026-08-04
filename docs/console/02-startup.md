# Starting the dashboard

Preferred Windows commands:

```bat
scripts\dashboard\start.cmd
scripts\dashboard\open.cmd
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/api/health
Invoke-RestMethod http://127.0.0.1:8765/api/build-info
```

Requires `.venv-training` and a built frontend `dashboard/frontend/dist`.
