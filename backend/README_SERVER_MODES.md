# Backend Server Modes

## 🚀 Production Mode (Recommended for Video Processing)

**Use this for stable video processing without interruptions:**

```bash
python backend/run_production.py
```

**Features:**
- ✅ No auto-restart (stable video processing)
- ✅ Models stay loaded in memory
- ✅ Processing won't be interrupted
- ❌ No auto-reload on code changes

**Best for:**
- Processing YouTube videos
- Long-running operations
- Production environments

---

## 🛠️ Development Mode (For Code Development)

**Use this when developing/debugging code:**

```bash
python backend/run_development.py
```

**Features:**
- ✅ Auto-reload on code changes
- ✅ Debug mode enabled
- ✅ Better error messages
- ❌ Will restart and interrupt video processing

**Best for:**
- Code development
- Debugging
- Testing new features

---

## ⚙️ Flexible Mode (Environment Variable Control)

**Use this with environment variables:**

```bash
# Production mode
FLASK_DEBUG=false python backend/run_app.py

# Development mode  
FLASK_DEBUG=true python backend/run_app.py
```

---

## 📊 Frontend Optimizations

The frontend now uses **intelligent polling**:

- **Active videos**: Polls every 10 seconds
- **Idle state**: Polls every 30 seconds  
- **Error state**: Polls every 30 seconds
- **Only polls active videos** (downloading, processing, resuming)

This reduces server load significantly!

---

## 🎯 Recommended Workflow

1. **For video processing**: Use `run_production.py`
2. **For development**: Use `run_development.py`
3. **Switch modes** as needed without losing data

**Note**: Always use production mode when processing videos to avoid interruptions! 