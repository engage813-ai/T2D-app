{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b0427256-367d-4bad-97d9-7b10c7c3c48d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn.metrics import classification_report, roc_auc_score\n",
    "import joblib\n",
    "import json\n",
    "import os\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "dc1281d7-34fc-412a-b870-964b29dac2fd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape: (7000, 14)\n",
      "Target distribution:\n",
      " Complication\n",
      "0    3784\n",
      "1    3216\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# Adjust the path if your file is somewhere else\n",
    "df = pd.read_excel(\"T2D dataset.xlsx\")\n",
    "\n",
    "target_col = \"Complication\"\n",
    "X = df.drop(columns=[target_col])\n",
    "y = df[target_col].astype(int)\n",
    "\n",
    "print(\"Shape:\", df.shape)\n",
    "print(\"Target distribution:\\n\", y.value_counts())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "49928b05-a8bb-46a8-8e2b-dfd0d28c2758",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Sex</th>\n",
       "      <th>BMI</th>\n",
       "      <th>HbA1c</th>\n",
       "      <th>Fasting  Glucose</th>\n",
       "      <th>LDL</th>\n",
       "      <th>HDL</th>\n",
       "      <th>Triglycerides</th>\n",
       "      <th>Systolic BP</th>\n",
       "      <th>Diastolic BP</th>\n",
       "      <th>Family History</th>\n",
       "      <th>Diet Quality</th>\n",
       "      <th>Duration Since Diagnosis</th>\n",
       "      <th>Complication</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>68</td>\n",
       "      <td>Female</td>\n",
       "      <td>36.1</td>\n",
       "      <td>8.4</td>\n",
       "      <td>121.2</td>\n",
       "      <td>44.1</td>\n",
       "      <td>25.0</td>\n",
       "      <td>241.6</td>\n",
       "      <td>125</td>\n",
       "      <td>84</td>\n",
       "      <td>1</td>\n",
       "      <td>Poor</td>\n",
       "      <td>18</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>81</td>\n",
       "      <td>Male</td>\n",
       "      <td>32.1</td>\n",
       "      <td>7.7</td>\n",
       "      <td>90.2</td>\n",
       "      <td>106.0</td>\n",
       "      <td>35.4</td>\n",
       "      <td>203.2</td>\n",
       "      <td>155</td>\n",
       "      <td>80</td>\n",
       "      <td>1</td>\n",
       "      <td>Average</td>\n",
       "      <td>21</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>58</td>\n",
       "      <td>Male</td>\n",
       "      <td>41.8</td>\n",
       "      <td>4.6</td>\n",
       "      <td>85.2</td>\n",
       "      <td>100.7</td>\n",
       "      <td>52.2</td>\n",
       "      <td>179.8</td>\n",
       "      <td>134</td>\n",
       "      <td>76</td>\n",
       "      <td>0</td>\n",
       "      <td>Average</td>\n",
       "      <td>13</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>44</td>\n",
       "      <td>Female</td>\n",
       "      <td>20.2</td>\n",
       "      <td>10.6</td>\n",
       "      <td>131.4</td>\n",
       "      <td>147.1</td>\n",
       "      <td>55.0</td>\n",
       "      <td>203.0</td>\n",
       "      <td>117</td>\n",
       "      <td>93</td>\n",
       "      <td>0</td>\n",
       "      <td>Poor</td>\n",
       "      <td>14</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>72</td>\n",
       "      <td>Male</td>\n",
       "      <td>25.5</td>\n",
       "      <td>7.9</td>\n",
       "      <td>71.5</td>\n",
       "      <td>96.2</td>\n",
       "      <td>55.3</td>\n",
       "      <td>208.4</td>\n",
       "      <td>173</td>\n",
       "      <td>99</td>\n",
       "      <td>0</td>\n",
       "      <td>Average</td>\n",
       "      <td>15</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Age     Sex   BMI  HbA1c  Fasting  Glucose    LDL   HDL  Triglycerides  \\\n",
       "0   68  Female  36.1    8.4             121.2   44.1  25.0          241.6   \n",
       "1   81    Male  32.1    7.7              90.2  106.0  35.4          203.2   \n",
       "2   58    Male  41.8    4.6              85.2  100.7  52.2          179.8   \n",
       "3   44  Female  20.2   10.6             131.4  147.1  55.0          203.0   \n",
       "4   72    Male  25.5    7.9              71.5   96.2  55.3          208.4   \n",
       "\n",
       "   Systolic BP  Diastolic BP  Family History Diet Quality  \\\n",
       "0          125            84               1         Poor   \n",
       "1          155            80               1      Average   \n",
       "2          134            76               0      Average   \n",
       "3          117            93               0         Poor   \n",
       "4          173            99               0      Average   \n",
       "\n",
       "   Duration Since Diagnosis  Complication  \n",
       "0                        18             1  \n",
       "1                        21             1  \n",
       "2                        13             0  \n",
       "3                        14             0  \n",
       "4                        15             1  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "9c56d71d-c483-4618-9679-95d539e37a46",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Numeric: ['Age', 'BMI', 'HbA1c', 'Fasting  Glucose', 'LDL', 'HDL', 'Triglycerides', 'Systolic BP', 'Diastolic BP', 'Family History', 'Duration Since Diagnosis']\n",
      "Categorical: ['Sex', 'Diet Quality']\n"
     ]
    }
   ],
   "source": [
    "numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()\n",
    "categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()\n",
    "print(\"Numeric:\", numeric_features)\n",
    "print(\"Categorical:\", categorical_features)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "36445e88-04ff-45a3-afd0-be3260f1b7a1",
   "metadata": {},
   "outputs": [],
   "source": [
    "numeric_transformer = Pipeline([\n",
    "    (\"imputer\", SimpleImputer(strategy=\"median\")),\n",
    "    (\"scaler\", StandardScaler())\n",
    "])\n",
    "\n",
    "categorical_transformer = Pipeline([\n",
    "    (\"imputer\", SimpleImputer(strategy=\"most_frequent\")),\n",
    "    (\"onehot\", OneHotEncoder(handle_unknown=\"ignore\"))\n",
    "])\n",
    "\n",
    "preprocessor = ColumnTransformer([\n",
    "    (\"num\", numeric_transformer, numeric_features),\n",
    "    (\"cat\", categorical_transformer, categorical_features)\n",
    "])\n",
    "\n",
    "pipeline = Pipeline([\n",
    "    (\"preprocessor\", preprocessor),\n",
    "    (\"clf\", MLPClassifier(max_iter=1000, random_state=42))\n",
    "])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "c36a00b7-736b-49ae-8d22-03a8bb854e99",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC AUC: 0.9970107919655019\n",
      "Classification report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.97      0.97      0.97       757\n",
      "           1       0.97      0.97      0.97       643\n",
      "\n",
      "    accuracy                           0.97      1400\n",
      "   macro avg       0.97      0.97      0.97      1400\n",
      "weighted avg       0.97      0.97      0.97      1400\n",
      "\n"
     ]
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, stratify=y, random_state=42\n",
    ")\n",
    "\n",
    "pipeline.fit(X_train, y_train)\n",
    "\n",
    "y_pred = pipeline.predict(X_test)\n",
    "y_proba = pipeline.predict_proba(X_test)[:, 1]\n",
    "\n",
    "print(\"ROC AUC:\", roc_auc_score(y_test, y_proba))\n",
    "print(\"Classification report:\\n\", classification_report(y_test, y_pred))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ecacd904-83f9-4db1-a214-3df2eadc938f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Model saved at models/mlp_pipeline.joblib\n"
     ]
    }
   ],
   "source": [
    "os.makedirs(\"models\", exist_ok=True)\n",
    "joblib.dump(pipeline, \"models/mlp_pipeline.joblib\")\n",
    "print(\"✅ Model saved at models/mlp_pipeline.joblib\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "432362be-ef86-4982-bf19-da7e0cdb928c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Features info saved at models/features_info.json\n"
     ]
    }
   ],
   "source": [
    "features_info = {}\n",
    "\n",
    "for col in numeric_features:\n",
    "    features_info[col] = {\n",
    "        \"median\": float(df[col].median()),\n",
    "        \"min\": float(df[col].min()),\n",
    "        \"max\": float(df[col].max())\n",
    "    }\n",
    "\n",
    "for col in categorical_features:\n",
    "    values = df[col].dropna().unique().tolist()\n",
    "    features_info[col] = {\"values\": [str(v) for v in values]}\n",
    "\n",
    "with open(\"models/features_info.json\", \"w\") as f:\n",
    "    json.dump(features_info, f)\n",
    "\n",
    "print(\"✅ Features info saved at models/features_info.json\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
