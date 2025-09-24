from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("Rule-Based Model Metrics:")
print("Accuracy:", accuracy_score(test_labels, y_pred_rule_based))
print("Precision:", precision_score(test_labels, y_pred_rule_based, average='weighted'))
print("Recall:", recall_score(test_labels, y_pred_rule_based, average='weighted'))
print("F1-score:", f1_score(test_labels, y_pred_rule_based, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(test_labels, y_pred_rule_based))
print("Report:\n", classification_report(test_labels, y_pred_rule_based))

print("Deep Learning Model Metrics:")
print("Accuracy:", accuracy_score(test_labels, y_pred_deep))
