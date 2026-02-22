# Quick Start Guide - Model Training and Testing

## Step-by-Step Instructions

### Step 1: Create Master CSV File
Combine all feature files into one training dataset:

```bash
python create_master_csv.py
```

**Output:** `model_training.csv`

**What it does:**
- Reads all `*_engine_features.csv` files from 4 folders
- Combines them with proper labels (Healthy, Misfire, MAP, Air_Leak)
- Creates a single CSV file for training

---

### Step 2: Train Models
Test multiple ML models and save the best one:

```bash
python ALLiswell.py --csv model_training.csv --save
```

**Output Files:**
- `best_engine_model.pkl` - Best performing model
- `best_engine_scaler.pkl` - Feature scaler
- `label_encoder.pkl` - Label encoder

**What it does:**
- Tests 4 models: Random Forest, SVM, XGBoost, Logistic Regression
- Compares their performance
- Selects the best model
- Saves all necessary files

**Expected Output:**
```
Model Comparison Summary
========================
Best Model: Random Forest
Accuracy: 0.9217 (92.17%)
```

---

### Step 3: Test Model Accuracy
Evaluate the trained model on training and testing data:

```bash
python test_model_accuracy.py --csv model_training.csv --model best_engine_model.pkl
```

**What it shows:**
- Overall accuracy (training vs testing)
- **Per-class accuracy** for each fault type:
  - Healthy
  - Misfire
  - MAP
  - Air_Leak
- Precision, Recall, F1-Score for each class
- Overfitting detection

**With Visualizations:**
```bash
python test_model_accuracy.py --csv model_training.csv --model best_engine_model.pkl --plot
```

**Save Results:**
```bash
python test_model_accuracy.py --csv model_training.csv --model best_engine_model.pkl --save-results
```

---

## Complete Workflow

```bash
# 1. Create master CSV
python create_master_csv.py

# 2. Train models and save best one
python ALLiswell.py --csv model_training.csv --save

# 3. Test model accuracy (with per-class metrics)
python test_model_accuracy.py --csv model_training.csv --model best_engine_model.pkl --plot --save-results
```

---

## Troubleshooting

### Error: "Model file not found"
**Solution:** You need to train the model first:
```bash
python ALLiswell.py --csv model_training.csv --save
```

### Error: "CSV file not found"
**Solution:** Create the master CSV first:
```bash
python create_master_csv.py
```

### Error: "No module named 'xgboost'"
**Solution:** Install xgboost (optional, script works without it):
```bash
pip install xgboost
```

---

## Understanding the Results

### Overall Metrics
- **Accuracy**: Overall correctness (how many predictions were correct)
- **Precision**: How many of the predicted positives were actually positive
- **Recall**: How many of the actual positives were found
- **F1-Score**: Balance between precision and recall

### Per-Class Accuracy
Shows how well the model detects each specific fault type:
- **Healthy**: How accurately healthy engines are identified
- **Misfire**: How accurately misfire faults are detected
- **MAP**: How accurately MAP sensor issues are detected
- **Air_Leak**: How accurately air leak faults are detected

### Status Indicators
- ✅ **Excellent** (≥90%): Very good detection
- ✅ **Good** (≥80%): Good detection
- ⚠️ **Fair** (≥70%): Acceptable but could improve
- ⚠️ **Poor** (≥60%): Needs improvement
- ❌ **Very Poor** (<60%): Significant improvement needed

---

## Files Created

After running all steps, you'll have:

1. **model_training.csv** - Master training dataset
2. **best_engine_model.pkl** - Trained model
3. **best_engine_scaler.pkl** - Feature scaler
4. **label_encoder.pkl** - Label encoder
5. **model_accuracy_comparison.png** - Visualization (if --plot used)
6. **accuracy_test_results.txt** - Detailed results (if --save-results used)

---

## Next Steps

After testing, you can:
1. Use the model in Streamlit app (automatically detected)
2. Fine-tune hyperparameters if accuracy is low
3. Collect more data for poorly performing classes
4. Retrain with different models

---

## Example Output

```
PER-CLASS ACCURACY SUMMARY - TESTING SET
============================================================
Class               Accuracy          Status              
------------------------------------------------------------
Healthy             95.00%            ✅ Excellent        
Misfire             92.00%            ✅ Excellent        
MAP                 88.00%             ✅ Good             
Air_Leak            90.00%             ✅ Excellent        
```

This shows your model can accurately detect each fault type!
