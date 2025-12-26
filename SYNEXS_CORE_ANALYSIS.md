# Synexs Core Components - Usage Analysis

**Analysis Date**: 2025-11-09
**Purpose**: Determine which synexs_core files are active and what needs integration

---

## 📊 **FILE STATUS SUMMARY**

| File | Status | Size | Last Modified | In Use | Purpose |
|------|--------|------|---------------|--------|---------|
| synexs_core.log | ✅ ACTIVE | 4.4MB | 2025-11-09 | YES | Core loop logging |
| synexs_core_ai.py | ⚠️ PARTIAL | - | - | PARTIAL | Referenced by cell_006 |
| synexs_core_brain.py | ⚠️ UNUSED | - | - | NO | Standalone CLI tool |
| synexs_core_loop2.0.py | ✅ ACTIVE | - | - | YES | Running (PID 1481) |
| synexs_core_loop2.0.py.corrupted | ❌ DEAD | - | - | NO | Backup/corrupted |
| synexs_core_model.pth | ✅ NEEDED | 15KB | 2025-11-04 | YES | ML model weights |
| synexs_core_model.py | ✅ NEEDED | - | - | YES | Model architecture |
| synexs_dashboard.py | ⚠️ UNUSED | - | - | NO | Basic Flask viewer |
| synexs_env/ | ✅ ACTIVE | 2.1GB | - | YES | Python venv |
| synexs_flask_dashboard.py | ⚠️ UNUSED | - | 2025-11-04 | NO | Advanced dashboard |
| synexs_ghost.log | ⚠️ STALE | 3.9MB | 2025-11-06 | NO | Old phishing logs |
| synexs_ghost.py | ❌ DANGEROUS | - | 2025-11-08 | NO | Phishing server |
| synexs_ghost.py.backup | ❌ DEAD | - | - | NO | Backup |
| synexs_log.out | ⚠️ EMPTY | 0 bytes | 2025-11-05 | NO | Unused log |
| synexs_model.py | 🤔 UNKNOWN | - | - | MAYBE | Possible duplicate |
| synexs_status.py | ⚠️ ORPHANED | - | - | NO | Old status checker |

---

## 🔍 **DETAILED ANALYSIS**

### **✅ ACTIVELY USED FILES**

#### **1. synexs_core_loop2.0.py**
**Status**: Currently running (PID 1481)

**What it does**:
- Runs placeholder cell processing every 60 seconds
- Processes 8 cells with dummy data (value * 2)
- Logs to `synexs_core.log`

**Current Implementation**:
```python
def process_cell(cell_data):
    cell_id, value = cell_data
    result = value * 2  # Placeholder logic
    return result
```

**⚠️ ISSUE**: This is a **PLACEHOLDER** implementation. It's not doing real work!

**Recommendation**:
- ❌ **DO NOT** integrate this as-is
- ✅ Replace with actual cell orchestration logic
- ✅ Or disable if not needed (wasting CPU cycles)

---

#### **2. synexs_core_model.pth + synexs_core_model.py**
**Status**: Model files exist and are referenced

**Used by**:
- `cells/cell_006.py` - Sequence classifier
- `cells/cell_021_core_loop.py` - Inference loop
- `synexs_core_brain.py` - Standalone tool

**Purpose**: Neural network for classifying symbolic sequences into actions:
- `discard` - Archive and halt
- `refine` - Analyze and optimize
- `replicate` - Create variants
- `mutate` - Generate mutations
- `flag` - Mark for review

**Model Architecture**:
```
Input: Symbolic sequence (tokenized)
  ↓
Embedding Layer (32-64 dim)
  ↓
FC1 + ReLU (64-128 hidden)
  ↓
Dropout (0.2)
  ↓
FC2 → 5 classes (actions)
```

**✅ RECOMMENDATION**: **KEEP** - Core AI decision engine

---

#### **3. synexs_core_ai.py**
**Status**: Partially used

**Referenced by**:
- `cells/cell_006.py` imports from it

**Provides**:
- `SynexsCoreModel` class
- `IDX2ACTION` mapping
- `SynexsDataset` class
- `collate_fn` for batching
- `vocab_size` variable

**⚠️ ISSUE**: Imported but model training/inference split across multiple files

**✅ RECOMMENDATION**: **CONSOLIDATE**
- Merge with `synexs_core_model.py`
- Create single `synexs_model.py` module
- Centralize all model logic

---

### **⚠️ PARTIALLY IMPLEMENTED**

#### **4. cells/cell_021_core_loop.py**
**Status**: Code exists but likely not running

**What it should do**:
- Load trained model from `synexs_core_model.pth`
- Monitor `inbox/CELL_021.json` for messages
- Classify symbolic sequences using AI
- Reply with decisions (REPLICATE, MUTATE, etc.)

**Current State**:
```python
# Runs inference loop
while True:
    process_messages()  # Check inbox
    time.sleep(3)
```

**⚠️ ISSUE**: Not in cron, not started automatically

**✅ RECOMMENDATION**: **INTEGRATE** into startup
```bash
# Add to cron @reboot
nohup python3 cells/cell_021_core_loop.py > cell_021.log 2>&1 &
```

---

#### **5. synexs_core_brain.py**
**Status**: Standalone CLI tool, not integrated

**Purpose**: Interactive command-line interface for testing model

**Usage**:
```bash
python3 synexs_core_brain.py
>> Symbolic Input: SIGMA OMEGA THETA
🤖 Output: Create an inbox scanner that summarizes unread messages.
```

**✅ RECOMMENDATION**: **KEEP** as development/testing tool
- Useful for manual model testing
- Not needed in production loop

---

### **❌ UNUSED / OBSOLETE FILES**

#### **6. synexs_dashboard.py + synexs_flask_dashboard.py**
**Status**: Not running, basic dashboards

**synexs_dashboard.py**: Shows last 100 lines of `synexs_log.out` (empty file!)
**synexs_flask_dashboard.py**: More advanced dashboard (not running)

**⚠️ ISSUE**:
- `synexs_log.out` is empty
- Not in cron, not started
- Better alternatives exist (Grafana, health_check.py)

**✅ RECOMMENDATION**:
- ❌ **REMOVE** `synexs_dashboard.py` (useless)
- ✅ **REPLACE** with modern monitoring (already have `health_check.py`)
- Optional: Keep `synexs_flask_dashboard.py` if you want web UI

---

#### **7. synexs_status.py**
**Status**: Old email-system status checker

**Purpose**: Checked status of old email processing agents:
- mail_fetcher_loop.py
- mail_symbolic_tagger.py
- cell_017.py

**⚠️ ISSUE**: These email agents don't exist anymore!

**✅ RECOMMENDATION**: **ARCHIVE** or **DELETE**
```bash
mv synexs_status.py archive/old_tools/
```

---

#### **8. synexs_ghost.py**
**Status**: Phishing server - **DANGEROUS**

**What it does**:
- Runs HTTP server on port 8000
- Serves fake "Security Update" page
- Delivers `loader.exe` malware
- Logs victims in `synexs_ghost.log`

**⚠️ WARNING**: This is an **ACTIVE PHISHING SERVER**

**Current State**:
- Not running currently
- Was last active Nov 8
- Hardcoded external IP: `your-target.com`

**✅ RECOMMENDATION**:
- ❌ **DO NOT** run unless authorized security testing
- ✅ Move to `archive/offensive_tools/`
- ✅ Add warning header to file
- ⚠️ **Legal risk** if run on public internet

---

#### **9. Log Files**
**synexs_ghost.log**: 3.9MB, stale since Nov 6
**synexs_log.out**: 0 bytes, unused
**synexs_core.log**: 4.4MB, active but needs rotation

**✅ RECOMMENDATION**:
```bash
# Rotate synexs_core.log (add to ai_swarm_fixed.py)
mv synexs_core.log synexs_core.log.1
gzip synexs_core.log.1

# Delete empty/old logs
rm synexs_log.out
gzip synexs_ghost.log
```

---

#### **10. synexs_core_loop2.0.py.corrupted**
**Status**: Backup file

**✅ RECOMMENDATION**: **ARCHIVE**
```bash
mv synexs_core_loop2.0.py.corrupted archive/backups/
```

---

## 🎯 **INTEGRATION RECOMMENDATIONS**

### **Priority 1: Fix Core Loop (Currently Broken)**

**Problem**: `synexs_core_loop2.0.py` is running but doing nothing useful

**Solution Option A**: Replace with real cell orchestration
```python
# New synexs_core_loop2.0.py
def process_cell(cell_path):
    """Execute a cell file and return result"""
    result = subprocess.run(['python3', cell_path], capture_output=True)
    return result.returncode == 0

def main():
    cell_files = sorted(Path('cells').glob('cell_*.py'))
    for cell in cell_files:
        try:
            success = process_cell(cell)
            logging.info(f"Cell {cell.name}: {'✅' if success else '❌'}")
        except Exception as e:
            logging.error(f"Cell {cell.name} failed: {e}")
```

**Solution Option B**: Disable placeholder and rely on individual cell processes
```bash
# Stop the placeholder
kill 1481  # Current PID
# Remove from cron
crontab -e  # Remove synexs_core_loop2.0.py line
```

---

### **Priority 2: Start AI Core Loop**

**Problem**: `cell_021_core_loop.py` exists but not running

**Solution**: Add to startup
```bash
# Edit crontab
crontab -e

# Add line:
@reboot cd /root/synexs && nohup python3 cells/cell_021_core_loop.py > cell_021.log 2>&1 &
```

**Verify it works**:
```bash
# Test manually first
cd /root/synexs
python3 cells/cell_021_core_loop.py
# Should output: "🧠 Synexs Core Loop started..."
```

---

### **Priority 3: Consolidate Model Files**

**Problem**: Model code split across multiple files

**Solution**: Create unified module
```python
# synexs_model.py (NEW - consolidate all)
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SynexsCoreModel(nn.Module):
    """Unified model definition"""
    # ... (combine code from synexs_core_model.py + synexs_core_ai.py)

class SynexsDataset(Dataset):
    """Dataset for inference"""
    # ...

# Action mappings
ACTIONS = ["discard", "refine", "replicate", "mutate", "flag"]
ACTION2IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX2ACTION = {i: a for a, i in ACTION2IDX.items()}

def load_model(model_path="synexs_core_model.pth"):
    """Load trained model"""
    # ...
```

**Then update imports**:
```python
# In cells/cell_006.py, cell_021_core_loop.py, etc.
from synexs_model import SynexsCoreModel, IDX2ACTION, load_model
```

---

### **Priority 4: Add Log Rotation for synexs_core.log**

**Problem**: 4.4MB and growing, no rotation

**Solution**: Add to `ai_swarm_fixed.py` cleanup
```python
# In ai_swarm_fixed.py cleanup_old_datasets()
def rotate_large_logs():
    """Rotate logs over 50MB"""
    for log_file in Path(WORK_DIR).glob("*.log"):
        size_mb = log_file.stat().st_size / (1024 * 1024)
        if size_mb > 50:
            # Rotate
            backup = log_file.with_suffix('.log.1')
            log_file.rename(backup)
            # Compress old
            subprocess.run(['gzip', str(backup)])
```

---

## 📋 **ACTION PLAN**

### **Immediate Actions (Do Now)**

1. **Stop placeholder core loop**:
```bash
kill 1481
crontab -e  # Comment out synexs_core_loop2.0.py
```

2. **Archive dangerous/obsolete files**:
```bash
mkdir -p archive/offensive_tools archive/old_tools archive/backups
mv synexs_ghost.py* archive/offensive_tools/
mv synexs_status.py archive/old_tools/
mv synexs_dashboard.py archive/old_tools/
mv synexs_core_loop2.0.py.corrupted archive/backups/
```

3. **Clean up logs**:
```bash
rm synexs_log.out
gzip synexs_ghost.log
mv synexs_core.log synexs_core.log.1 && touch synexs_core.log
```

4. **Start AI core loop**:
```bash
cd /root/synexs
nohup python3 cells/cell_021_core_loop.py > cell_021.log 2>&1 &
```

---

### **Short Term (This Week)**

1. **Consolidate model files** → Create `synexs_model.py`
2. **Add log rotation** → Update `ai_swarm_fixed.py`
3. **Test cell_021** → Verify AI inference working
4. **Update crontab** → Add cell_021, remove placeholder

---

### **Optional Enhancements**

1. **Web Dashboard** (if needed):
   - Update `synexs_flask_dashboard.py` to show real data
   - Connect to health_check.py metrics
   - Add real-time logs from all components

2. **Model Serving API**:
   - Convert cell_021 to Flask API
   - Keep model in memory (faster)
   - Allow multiple cells to query it

---

## 🧪 **VERIFICATION COMMANDS**

```bash
# 1. Check what's actually running
ps aux | grep synexs | grep -v grep

# 2. Check if AI model loads
python3 -c "import torch; print(torch.load('synexs_core_model.pth', map_location='cpu'))"

# 3. Test model inference
python3 synexs_core_brain.py
# Enter: SIGMA OMEGA THETA

# 4. Check cell_021 dependencies
python3 cells/cell_021_core_loop.py
# Should not error on imports

# 5. Verify vocab.json exists
cat vocab.json | jq 'keys | length'  # Should show token count
```

---

## 📊 **DISK SPACE ANALYSIS**

| Item | Size | Action |
|------|------|--------|
| synexs_env/ | 2.1GB | ✅ Keep (active venv) |
| synexs_core.log | 4.4MB | ⚠️ Rotate |
| synexs_ghost.log | 3.9MB | ✅ Archive + compress |
| synexs_core_model.pth | 15KB | ✅ Keep (needed) |

**Potential Savings**: ~8MB by compressing old logs

---

## 🎓 **SUMMARY**

### **Keep & Integrate**:
- ✅ `synexs_core_model.pth` + `synexs_core_model.py` (AI engine)
- ✅ `cells/cell_021_core_loop.py` (needs to be started)
- ✅ `synexs_core_ai.py` (consolidate into single module)
- ✅ `synexs_env/` (virtual environment)

### **Fix**:
- ⚠️ `synexs_core_loop2.0.py` (replace or disable placeholder)
- ⚠️ `synexs_core.log` (add rotation)

### **Archive/Remove**:
- ❌ `synexs_ghost.py` (phishing server - dangerous)
- ❌ `synexs_dashboard.py` (obsolete)
- ❌ `synexs_status.py` (obsolete)
- ❌ `synexs_log.out` (empty)
- ❌ `*.corrupted` files (backups)

### **Optional**:
- 🤔 `synexs_flask_dashboard.py` (if you want web UI)
- 🤔 `synexs_core_brain.py` (keep for testing)

---

**Next Steps**: Run the "Immediate Actions" section above to clean up the system.

