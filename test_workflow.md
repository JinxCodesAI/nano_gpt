# Protect Edits - Corrected Workflow Test

## Expected User Flow:
1. ✅ Open application 
2. ✅ Select model
3. ✅ Click "Start Generate"
4. ✅ After generation completed, mark "Protect Edits" 
5. ✅ Go to step 3/20 (Editable)
6. ✅ Edit fragment deleting some masks and typing "JOHNNY:"
7. ✅ Click "Next>" → **Should save edits and protect "JOHNNY:" positions**
8. ✅ Click "Previous" → **Should show "JOHNNY:" text still there**

## Key Improvements Made:

### ❌ **Previous (Broken) Behavior:**
- Saved on every keystroke → inefficient, wrong timing
- Compared against previous step → missed user edits
- Static protection set → couldn't update during generation

### ✅ **New (Correct) Behavior:**  
- **Save only on navigation**: Previous/Next buttons, Restart
- **Compare against baseline**: Original model output vs current text
- **Dynamic protection updates**: InferenceRunner gets updated positions in real-time

## Implementation Details:

### **1. Navigation-Based Saving:**
```python
def on_substep_changed(self, substep_key):
    # Before switching, save any edits from current step if protection enabled
    if self.current_substep_key and self.protect_edits_enabled:
        self.save_current_edits()  # NEW: Only save on navigation
```

### **2. Proper Edit Detection:**
```python
def save_current_edits(self):
    # Compare against baseline (original model output) not previous step
    baseline_tokens = self.baseline_tokens[self.current_substep_key]  # FIXED
    changed_positions = torch.where(
        (baseline_tokens[0] != modified_tokens[0]) &  # Changed from baseline
        (modified_tokens[0] != mask_token_id)         # And new token is not mask
    )[0]
```

### **3. Baseline Storage:**
```python
def on_substep_complete(self, step_num, substep_type, tokens, is_editable):
    # Store baseline tokens (original model output before user edits)
    self.baseline_tokens[substep_key] = tokens.clone()  # NEW: Store original
```

### **4. Dynamic Protection Updates:**
```python
# Update the InferenceRunner with new protected positions
if self.inference_runner and hasattr(self.inference_runner, 'update_protected_positions'):
    self.inference_runner.update_protected_positions(self.protected_positions)
```

## Expected Debug Output:
When user types "JOHNNY:" and clicks Next:
```
DEBUG: save_current_edits - saving edits for 3_remasking
DEBUG: Protected 7 newly edited positions: [45, 46, 47, 48, 49, 50, 51]
DEBUG: Total protected positions: [45, 46, 47, 48, 49, 50, 51]
DEBUG: InferenceRunner - updated protected positions: [45, 46, 47, 48, 49, 50, 51]
```

When apply_remasking runs:
```
DEBUG: apply_remasking called with protected_positions: [45, 46, 47, 48, 49, 50, 51]
DEBUG: batch 0: unmasked=150, protected=7, remaskable=143
DEBUG: filtered out 7 protected positions
```

## Test Instructions:
1. Follow the user flow above
2. Check debug output matches expectations
3. Verify "JOHNNY:" stays protected with green highlighting
4. Verify navigation preserves edits correctly