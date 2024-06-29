import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import threading




def load_dataset():
   global dataset
   file_path = filedialog.askopenfilename()
   if file_path:
       try:
           dataset = pd.read_csv(file_path)
           run_button.config(state=tk.NORMAL)
           messagebox.showinfo("Dataset Loaded", "Dataset loaded successfully!")
       except Exception as e:
           messagebox.showerror("Error", f"Failed to load dataset: {e}")




def run_tpot():
   global pipeline, X_test_split, y_test_split, y_pred
   if dataset is not None:
       try:
           # Assume the last column is the target variable
           X = dataset.iloc[:, :-1]
           y = dataset.iloc[:, -1]


           # Encode target labels if they are categorical
           if y.dtype == 'object':
               le = LabelEncoder()
               y = le.fit_transform(y)


           X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X, y, test_size=0.2,
                                                                                       random_state=42)


           tpot = TPOTClassifier(verbosity=2, generations=5, population_size=20, random_state=42)
           tpot.fit(X_train_split, y_train_split)


           y_pred = tpot.predict(X_test_split)
           accuracy = accuracy_score(y_test_split, y_pred)
           result_label.config(text=f"Test Accuracy: {accuracy:.4f}")


           pipeline = tpot.fitted_pipeline_
           export_button.config(state=tk.NORMAL)


           messagebox.showinfo("Optimization Complete", "TPOT optimization complete!")
       except Exception as e:
           messagebox.showerror("Error", f"Failed to run TPOT optimization: {e}")




def run_tpot_thread():
   threading.Thread(target=run_tpot).start()




def export_predictions():
   if y_pred is not None:
       file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
       if file_path:
           try:


               results_df = pd.DataFrame(X_test_split)
               results_df['True_Label'] = y_test_split
               results_df['Predicted_Label'] = y_pred


               # Save the DataFrame to a CSV file
               results_df.to_csv(file_path, index=False)
               messagebox.showinfo("Predictions Exported", "Predictions exported successfully!")
           except Exception as e:
               messagebox.showerror("Error", f"Failed to export predictions: {e}")




if __name__ == "__main__":
   # Initialize global variables
   dataset = None
   pipeline = None
   X_test_split, y_test_split, y_pred = None, None, None


   # Create the main window
   root = tk.Tk()
   root.title("AutoML with TPOT - The Pycodes")
   root.geometry("400x250")


   # Create and place widgets
   label = tk.Label(root, text="AutoML with TPOT", font=("Helvetica", 16))
   label.pack(pady=10)


   load_button = tk.Button(root, text="Load Dataset", command=load_dataset)
   load_button.pack(pady=10)


   run_button = tk.Button(root, text="Run TPOT Optimization", command=run_tpot_thread, state=tk.DISABLED)
   run_button.pack(pady=10)


   result_label = tk.Label(root, text="", font=("Helvetica", 12))
   result_label.pack(pady=10)


   export_button = tk.Button(root, text="Export Predictions", command=export_predictions, state=tk.DISABLED)
   export_button.pack(pady=10)


   # Start the Tkinter event loop
   root.mainloop()
