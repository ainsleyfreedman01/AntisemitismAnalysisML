import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score


def load_and_prepare(path: str):
	"""Load raw campus reports CSV and prepare a per-day aggregated DataFrame.

	Returns a DataFrame with columns: `Date` (datetime), `incident_count` (int),
	`has_protest` (0/1), temporal features, `is_holiday` (US or weekend fallback),
	`is_jewish_holiday` (1 if date is a Jewish holiday when detectable), and `holiday_type`.
	"""
	raw = pd.read_csv(path)

	# Try to find a date column; common header in this dataset is 'Date of Incident'
	date_col = None
	for candidate in ["Date of Incident", "date", "Date"]:
		if candidate in raw.columns:
			date_col = candidate
			break
	if date_col is None:
		raise ValueError("Could not find a date column in the CSV; expected 'Date of Incident' or similar.")

	raw["Date"] = pd.to_datetime(raw[date_col], errors="coerce")
	raw = raw.dropna(subset=["Date"])

	# Aggregate incidents per calendar date and detect whether any row is a Protest/Action
	incident_type_col = None
	for c in ["Incident Type", "incident_type", "IncidentType", "Incident"]:
		if c in raw.columns:
			incident_type_col = c
			break

	def _is_protest_series(s):
		# common label in dataset is 'Protest/Action'
		return s.str.contains('Protest', case=False, na=False).any()

	agg_map = {"Date": ("Date", 'first')}

	grouped = raw.groupby(raw["Date"].dt.date)
	daily = grouped.size().reset_index(name="incident_count")
	# compute has_protest if we have an incident type column
	if incident_type_col is not None:
		daily['has_protest'] = grouped[incident_type_col].apply(lambda s: int(_is_protest_series(s))).values
	else:
		daily['has_protest'] = 0

	# normalize column name and ensure datetime index
	daily["Date"] = pd.to_datetime(daily["Date"]) 
	daily = daily.sort_values("Date").reset_index(drop=True)

	# Day of week feature
	daily["day_of_week"] = daily["Date"].dt.day_name()

	# Reindex to a complete daily range so rolling windows include zeros for missing days
	daily = daily.set_index("Date")
	full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
	daily = daily.reindex(full_idx, fill_value=0).rename_axis("Date").reset_index()

	# Temporal features
	daily["day_of_week"] = daily["Date"].dt.day_name()
	daily["day_of_month"] = daily["Date"].dt.day
	daily["month"] = daily["Date"].dt.month
	daily["week_of_year"] = daily["Date"].dt.isocalendar().week.astype(int)
	daily["is_weekend"] = (daily["Date"].dt.weekday >= 5).astype(int)

	# Additional temporal features
	daily["day_of_year"] = daily["Date"].dt.dayofyear

	# EWMA of past incidents (shifted so it doesn't include current day)
	daily["incidents_ewm7"] = daily["incident_count"].shift(1).ewm(span=7, adjust=False).mean().fillna(0)

	# Rolling features: previous windows (shifted so they don't include current day)
	daily = daily.sort_values("Date").reset_index(drop=True)
	daily["incidents_last_7d"] = daily["incident_count"].rolling(window=7, min_periods=1).sum().shift(1).fillna(0)
	daily["incidents_last_30d"] = daily["incident_count"].rolling(window=30, min_periods=1).sum().shift(1).fillna(0)
	daily["incidents_prev_day"] = daily["incident_count"].shift(1).fillna(0)

	# Determine holidays: prefer the `holidays` package if available for US and Israel
	# and also try to detect Jewish holidays specifically
	daily["is_holiday"] = 0
	daily["holiday_type"] = ""
	daily["is_jewish_holiday"] = 0
	try:
		import holidays
		us_holidays = holidays.US()
		# Israel country holidays often include Jewish holidays; use it to detect Jewish holidays
		try:
			il_holidays = holidays.CountryHoliday('IL')
		except Exception:
			il_holidays = None

		def _is_hol(d):
			return 1 if d.date() in us_holidays else 0

		def _hol_name(d):
			return us_holidays.get(d.date(), "") or ""

		daily["is_holiday"] = daily["Date"].apply(_is_hol)
		daily["holiday_type"] = daily["Date"].apply(_hol_name)

		if il_holidays is not None:
			# mark jewish holiday if date appears in Israel holiday set
			daily["is_jewish_holiday"] = daily["Date"].apply(lambda d: 1 if d.date() in il_holidays else 0)

	except Exception:
		# fallback: leave is_holiday as weekend indicator later
		daily["is_holiday"] = (daily["Date"].dt.weekday >= 5).astype(int)
		# is_jewish_holiday remains 0 unless we can compute it below

	# If holidays package not available or Israel not present, try convertdate to map hebrew dates
	# Prefer pyluach for precise Hebrew calendar mapping; fall back to convertdate.hebrew or heuristics
	if daily['is_jewish_holiday'].sum() == 0:
			try:
				# pyluach gives direct access to hebrew calendar; detect multi-day holidays and diaspora second-days
				import pyluach

				def _is_jewish_pyluach(d):
					try:
						# convert to hebrew date object
						hy = pyluach.dates.GregorianDate(d.year, d.month, d.day).to_heb()
						hm, hd = hy.month, hy.day
						# Helper: check ranges safely
						def in_range(month, start, end):
							return hm == month and start <= hd <= end

						# Rosh Hashanah: Tishri 1-2
						if in_range(7, 1, 2):
							return 1
						# Yom Kippur: Tishri 10
						if in_range(7, 10, 10):
							return 1
						# Sukkot: Tishri 15-21 (first week), Shemini Atzeret/Simchat Torah: 22-23
						if (in_range(7, 15, 21) or in_range(7, 22, 23)):
							return 1
						# Hanukkah: Kislev 25 through Tevet ~2-3 (handle month boundary)
						if (hm == 9 and hd >= 25) or (hm == 10 and hd <= 3):
							return 1
						# Purim: Adar (Adar II in leap years) 14
						# pyluach uses month numbers where Adar II is 13 in leap years
						if (hm in (12, 13)) and hd == 14:
							return 1
						# Passover: Nisan 15-21 (diaspora may include 22)
						if (in_range(1, 15, 21) or in_range(1, 22, 22)):
							return 1
						# Shavuot: Sivan 6-7 (diaspora 2 days sometimes 6-7)
						if (in_range(3, 6, 7)):
							return 1
						# Tisha B'Av: Av 9 (month number 5 in pyluach mapping)
						if in_range(5, 9, 9):
							return 1
						return 0
					except Exception:
						return 0

				daily['is_jewish_holiday'] = daily['Date'].apply(_is_jewish_pyluach)
			except Exception:
				# try convertdate as fallback (similar logic)
				try:
					import convertdate.hebrew as hebrew

					def _is_jewish_convert(d):
						try:
							hy, hm, hd = hebrew.from_gregorian(d.year, d.month, d.day)
							# Rosh Hashanah: 7/1-2; YK 7/10; Sukkot 7/15-21; Shemini Atzeret/Simchat 7/22-23
							if (hm, hd) in ((7, 1), (7, 2), (7, 10)):
								return 1
							if (hm == 7 and 15 <= hd <= 23):
								return 1
							# Hanukkah: Kislev 25 -> Tevet 2-3
							if (hm == 9 and hd >= 25) or (hm == 10 and hd <= 3):
								return 1
							# Purim: Adar 14 (handle Adar/Adar II via month numbers)
							if (hm in (12, 13)) and hd == 14:
								return 1
							# Passover: Nisan 15-21 (and local 22)
							if (hm == 1 and 15 <= hd <= 22):
								return 1
							# Shavuot: Sivan 6-7
							if (hm == 3 and 6 <= hd <= 7):
								return 1
							return 0
						except Exception:
							return 0

					daily['is_jewish_holiday'] = daily['Date'].apply(_is_jewish_convert)
				except Exception:
						# last fallback: leave zeros
							pass

	# If still no jewish holidays detected, use the user-provided hardcoded list
	if daily['is_jewish_holiday'].sum() == 0:
		try:
			from scripts.machine_learning_scripts.hardcoded_jewish_holidays import get_hardcoded_jewish_holidays
			holidays = get_hardcoded_jewish_holidays()
			# holidays is a list of {'date': date, 'name': name}
			hol_set = set([h['date'] for h in holidays])
			name_map = {h['date']: h['name'] for h in holidays}
			# mark days
			daily['is_jewish_holiday'] = daily['Date'].apply(lambda d: 1 if d.date() in hol_set else 0)
			# fill holiday_type for matches (append to existing holiday_type if present)
			def _map_holiday_name(d):
				name = name_map.get(d.date()) if d.date() in name_map else None
				if name:
					# if there was an existing holiday_type, append
					return name if pd.isna(daily_holiday_name_map.get(d.date(), pd.NA)) else name
				return ""
			# create a simple mapping of date->existing holiday_type to avoid repeated lookups
			daily_holiday_name_map = {row.Date.date(): row.holiday_type for row in daily.itertuples()}
			# update holiday_type where jewish holiday present
			daily.loc[daily['is_jewish_holiday'] == 1, 'holiday_type'] = daily.loc[daily['is_jewish_holiday'] == 1, 'Date'].apply(
				lambda d: name_map.get(d.date(), "Jewish holiday")
			)
		except Exception:
			# don't fail the pipeline if importing the helper fails
			pass

	return daily


if __name__ == "__main__":
	df = load_and_prepare("data/campus_reports.csv")

	# Ensure outputs directory exists for persisted artifacts
	os.makedirs('outputs', exist_ok=True)

	# --- Protest on Jewish holiday analysis + Fisher exact test ---
	# Only run if `has_protest` and `is_jewish_holiday` exist
	if 'has_protest' in df.columns and 'is_jewish_holiday' in df.columns:
		# contingency table: [[protests_on_jh, nonprotests_on_jh],[protests_not_jh, nonprotests_not_jh]]
		jh = df[df['is_jewish_holiday'] == 1]
		njh = df[df['is_jewish_holiday'] == 0]
		protests_on_jh = int(jh['has_protest'].sum()) if not jh.empty else 0
		nonprotests_on_jh = int(len(jh) - protests_on_jh) if not jh.empty else 0
		protests_not_jh = int(njh['has_protest'].sum()) if not njh.empty else 0
		nonprotests_not_jh = int(len(njh) - protests_not_jh) if not njh.empty else 0
		contingency = [[protests_on_jh, nonprotests_on_jh], [protests_not_jh, nonprotests_not_jh]]

		print('\n--- Protest vs Jewish Holiday Contingency ---')
		print('contingency (rows: [JewishHoliday, NotJewishHoliday], cols: [Protest, NoProtest])')
		print(contingency)

		# Save contingency and basic rates
		metrics = {
			'protests_on_jh': protests_on_jh,
			'nonprotests_on_jh': nonprotests_on_jh,
			'protests_not_jh': protests_not_jh,
			'nonprotests_not_jh': nonprotests_not_jh,
			'total_jh_days': len(jh),
			'total_not_jh_days': len(njh),
			'protest_rate_jh': (protests_on_jh / len(jh)) if len(jh) > 0 else np.nan,
			'protest_rate_not_jh': (protests_not_jh / len(njh)) if len(njh) > 0 else np.nan,
		}
		metrics_df = pd.DataFrame([metrics])
		metrics_df.to_csv(os.path.join('outputs', 'protest_jewish_holiday_metrics.csv'), index=False)

		# Try Fisher exact test (scipy)
		try:
			from scipy.stats import fisher_exact
			oddsratio, pvalue = fisher_exact(contingency)
			print('Fisher exact test p-value:', pvalue)
			metrics_df['fisher_pvalue'] = pvalue
			metrics_df['fisher_oddsratio'] = oddsratio
			metrics_df.to_csv(os.path.join('outputs', 'protest_jewish_holiday_metrics.csv'), index=False)
		except Exception as e:
			print('Could not run Fisher exact test (scipy missing?):', e)

		# Save a small bar plot of protest rates
		try:
			import matplotlib.pyplot as plt
			rates = df.groupby('is_jewish_holiday')['has_protest'].mean()
			fig, ax = plt.subplots()
			rates.plot(kind='bar', ax=ax)
			ax.set_title('Protest rate by is_jewish_holiday')
			ax.set_xlabel('is_jewish_holiday')
			ax.set_ylabel('Proportion of days with protest')
			plt.tight_layout()
			fig.savefig(os.path.join('outputs', 'protest_rate_by_jewish_holiday.png'))
			plt.close(fig)
		except Exception as e:
			print('Failed to save protest rate plot:', e)

	else:
		print('Columns `has_protest` or `is_jewish_holiday` not present — skipping protest/Holiday statistical test.')
	# Calculate average incidents on non-holidays (guard for empty selection)
	non_holiday = df[df["is_holiday"] == 0]
	if non_holiday.empty:
		average_incidents = df["incident_count"].mean()
	else:
		average_incidents = non_holiday["incident_count"].mean()

	# Create a binary target column
	df["is_high_incident_day"] = (df["incident_count"] > average_incidents).astype(int)

	# Select Features (ensure columns exist)
	feature_cols = [c for c in ["is_holiday", "day_of_week", "holiday_type"] if c in df.columns]
	X = df[feature_cols]
	y = df["is_high_incident_day"]

	# Encode categorical variables
	X = pd.get_dummies(X, drop_first=True)

	# Split Data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Train Random Forest Model
	def evaluate_and_print(name, model, X_tr, y_tr, X_te, y_te):
		model.fit(X_tr, y_tr)
		y_pred = model.predict(X_te)
		print(f"--- {name} ---")
		print("Accuracy:", accuracy_score(y_te, y_pred))
		print(classification_report(y_te, y_pred))
		try:
			from sklearn.metrics import confusion_matrix
			print("Confusion matrix:\n", confusion_matrix(y_te, y_pred))
		except Exception:
			pass

	# Baseline: default RandomForest
	baseline = RandomForestClassifier(n_estimators=100, random_state=42)
	evaluate_and_print("Baseline (default)", baseline, X_train, y_train, X_test, y_test)

	# Class-weighted RandomForest
	cw = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
	evaluate_and_print("Class-weighted (balanced)", cw, X_train, y_train, X_test, y_test)

	# Oversampling the minority class (try imblearn.SMOTE if available, fallback to simple repeat oversample)
	try:
		from imblearn.over_sampling import SMOTE
		sm = SMOTE(random_state=42)
		X_res, y_res = sm.fit_resample(X_train, y_train)
		sm_model = RandomForestClassifier(n_estimators=100, random_state=42)
		evaluate_and_print("SMOTE oversample", sm_model, X_res, y_res, X_test, y_test)
	except Exception:
		# simple random oversample by repetition
		minority_class = y_train.value_counts().idxmin()
		maj_class = y_train.value_counts().idxmax()
		n_maj = y_train.value_counts().max()
		X_min = X_train[y_train == minority_class]
		y_min = y_train[y_train == minority_class]
		# repeat minority rows to match majority count
		reps = int(np.ceil(n_maj / len(y_min))) if len(y_min) > 0 else 1
		X_min_rep = pd.concat([X_min] * reps, ignore_index=True)
		y_min_rep = pd.Series(list(y_min) * reps)  # type: ignore
		X_res = pd.concat([X_train[y_train == maj_class], X_min_rep], ignore_index=True)
		y_res = pd.concat([y_train[y_train == maj_class].reset_index(drop=True), y_min_rep.reset_index(drop=True)], ignore_index=True)
		# shuffle
		perm = np.random.permutation(len(y_res))
		X_res = X_res.reset_index(drop=True).iloc[perm].reset_index(drop=True)
		y_res = y_res.reset_index(drop=True).iloc[perm].reset_index(drop=True)

		os_model = RandomForestClassifier(n_estimators=100, random_state=42)
		evaluate_and_print("Simple oversample (repeat)", os_model, X_res, y_res, X_test, y_test)

	# Show prediction distribution for baseline model
	try:
		baseline_preds = baseline.predict(X_test)
		unique, counts = np.unique(baseline_preds, return_counts=True)
		print("Baseline prediction distribution:", dict(zip(unique, counts)))
	except Exception:
		pass

	# -------------------- Hyperparameter tuning + threshold sweep --------------------
	def hyperparameter_tuning_and_threshold_search(X_tr, y_tr, X_te, y_te):
		print('\n--- Hyperparameter tuning (class_weight & max_depth) ---')
		# expanded grid: tune number of trees and max_features as well
		param_grid = {
			'class_weight': [None, 'balanced'],
			'max_depth': [None, 3, 5, 10],
			'n_estimators': [100, 300],
			'max_features': ['sqrt', 'log2', 'auto']
		}
		cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
		base = RandomForestClassifier(n_estimators=100, random_state=42)
		gs = GridSearchCV(base, param_grid, scoring='f1', cv=cv, n_jobs=-1)
		gs.fit(X_tr, y_tr)
		print('Best params:', gs.best_params_)
		print('Best CV F1:', gs.best_score_)

		best = gs.best_estimator_
		# get probabilities for positive class
		if hasattr(best, 'predict_proba'):
			probs = best.predict_proba(X_te)[:, 1]
		else:
			# fallback to decision_function
			probs = best.decision_function(X_te)

		precisions, recalls, thresholds = precision_recall_curve(y_te, probs)
		f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
		best_idx = np.nanargmax(f1s)
		best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
		print(f'Best threshold by F1 on test: {best_threshold:.3f}, F1: {f1s[best_idx]:.3f}')

		# final predict using threshold
		y_pred_thresh = (probs >= best_threshold).astype(int)
		print('Metrics at best threshold:')
		print('Accuracy:', accuracy_score(y_te, y_pred_thresh))
		print(classification_report(y_te, y_pred_thresh))
		try:
			print('ROC AUC (probabilities):', roc_auc_score(y_te, probs))
		except Exception:
			pass

		# save best model for later reuse
		try:
			os.makedirs('models', exist_ok=True)
			joblib.dump(best, os.path.join('models', 'best_holiday_model.pkl'))
			print('Saved best model to models/best_holiday_model.pkl')
		except Exception as e:
			print('Failed to save model:', e)

		# Persist PR curve and F1 vs threshold plots to outputs/
		try:
			import matplotlib.pyplot as plt
			os.makedirs('outputs', exist_ok=True)
			# PR curve
			fig1, ax1 = plt.subplots()
			ax1.plot(recalls, precisions, label='PR curve')
			ax1.set_xlabel('Recall')
			ax1.set_ylabel('Precision')
			ax1.set_title('Precision-Recall Curve')
			ax1.legend()
			fig1.tight_layout()
			fig1.savefig(os.path.join('outputs', 'pr_curve_best_holiday_model.png'))
			plt.close(fig1)

			# F1 vs threshold (align lengths: thresholds has len = len(precisions)-1)
			if len(thresholds) > 0:
				fig2, ax2 = plt.subplots()
				ax2.plot(thresholds, f1s[:-1])
				ax2.set_xlabel('Threshold')
				ax2.set_ylabel('F1')
				ax2.set_title('F1 vs Decision Threshold')
				fig2.tight_layout()
				fig2.savefig(os.path.join('outputs', 'f1_vs_threshold_best_holiday_model.png'))
				plt.close(fig2)
		except Exception as e:
			print('Failed to save PR/F1 plots:', e)

		return gs.best_params_, gs.best_score_, best_threshold, best, (precisions, recalls, f1s)

	try:
		best_params, best_cv_f1, best_thresh, best_model, pr_data = hyperparameter_tuning_and_threshold_search(X_train, y_train, X_test, y_test)
	except Exception as e:
		print('Hyperparameter tuning failed:', e)
