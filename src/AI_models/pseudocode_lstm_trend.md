# LSTM Trend 系統 Pseudocode

## 主要流程

### 1. 訓練流程 (train_store_lstm)

```
FUNCTION train_store_lstm(store_id, raw_dataframe, parameters, callbacks, simulate_mode):
    
    IF simulate_mode == TRUE:
        CREATE empty_model_files
        WRITE fake_metrics
        UPDATE index_registry
        RETURN model_path, metrics_path, version
    
    // 資料前處理
    VALIDATE required_columns [ts, store_id, device_id, device_type, temp_current, t_room, defrost_status, power]
    
    processed_df = CALL _resample_minutely(raw_dataframe)
        FOR each (store_id, device_id) group:
            RESAMPLE to 1-minute intervals
            FORWARD_FILL missing values
            COMPUTE ingest_ts from defrost_status transitions
    
    FILTER_OUT aircon_devices  // 排除空調
    
    // 檢查資料充足性
    min_required_length = window + horizon
    FOR each device:
        IF device_data_length < min_required_length:
            REMOVE device from training_set
    
    IF no_valid_devices:
        THROW InsufficientDataError
    
    // 資料切分
    time_split_point = 80th_percentile(timestamps)
    train_data = data WHERE timestamp <= time_split_point
    validation_data = data WHERE timestamp > time_split_point
    
    // 建立 Dataset
    train_dataset = CREATE SeqSurvivalDataset(train_data, features, device_type_map, window, horizon, step)
    val_dataset = CREATE SeqSurvivalDataset(validation_data, features, device_type_map, window, horizon, step)
    
    // 模型訓練
    lstm_model = CREATE HazardLSTM(input_dim=len(features), hidden, layers, horizon)
    
    FOR epoch = 1 TO max_epochs:
        // 訓練階段
        FOR each batch IN train_dataset:
            X, y_hazard, mask = batch
            logits = lstm_model(X)
            loss = hazard_negative_log_likelihood(logits, y_hazard, mask)
            BACKPROPAGATE loss
        
        // 驗證階段
        validation_loss = EVALUATE model ON val_dataset
        
        IF validation_loss < best_validation_loss:
            SAVE best_model_state
    
    // 評估與儲存
    mae, hit_rate, calibration = EVALUATE_METRICS(lstm_model, val_dataset)
    
    SAVE model_weights TO model_path
    SAVE metrics TO metrics_path
    UPDATE index_registry(store_id, version, model_path, metrics_path)
    
    // 建立預測上下文快照
    context_data = EXTRACT last_window_plus_horizon_minutes(processed_df)
    SAVE context_data TO context_path
    
    RETURN model_path, metrics_path, version

END FUNCTION
```

### 2. 預測流程 (predict_store)

```
FUNCTION predict_store(store_id, horizon, simulate_mode, version):
    
    IF simulate_mode == TRUE:
        GENERATE fake_prediction_data
        RETURN prediction_file, meta_file, version, stats
    
    // 載入模型
    version, model_info = RESOLVE_VERSION_FROM_INDEX(store_id, version)
    lstm_model = LOAD_MODEL(model_info.model_path)
    
    // 載入預測上下文
    context_data = LOAD_CSV(context_path)
    
    predictions = []
    
    FOR each device IN context_data:
        device_group = GROUP_BY device_id
        
        IF device_data_length < required_window:
            CONTINUE  // 跳過資料不足的設備
        
        // 提取特徵窗口
        feature_window = EXTRACT_LAST_N_MINUTES(device_group, window_size)
        X = feature_window[features].TO_NUMPY()
        
        // 模型預測
        hazard_logits = lstm_model(X)
        ttw_quantiles = CONVERT_HAZARD_TO_QUANTILES(hazard_logits, [0.1, 0.5, 0.9])
        
        // 安全規則檢查
        current_temp = X[-1][temp_current_index]
        device_type = device_group[device_type].LAST()
        defrost_status = X[-1][defrost_status_index]
        ingest_time = device_group[ingest_ts].LAST()
        
        exclude_flag = FALSE
        exclude_reasons = []
        
        // 排除條件檢查
        IF defrost_status == 1:
            exclude_flag = TRUE
            exclude_reasons.APPEND("defrost")
        
        IF ingest_time < 10:
            exclude_flag = TRUE
            exclude_reasons.APPEND("recent_defrost")
        
        unloadable_time = MAX(FLOOR(ttw_quantiles.p10 - safety_buffer), 0)
        IF unloadable_time < 10:
            exclude_flag = TRUE
            exclude_reasons.APPEND("too_short")
        
        IF IS_AIRCON(device_type):
            exclude_flag = TRUE
            exclude_reasons.APPEND("aircon")
        
        near_limit = GET_NEAR_LIMIT_BY_TYPE(device_type)
        IF current_temp >= near_limit:
            exclude_flag = TRUE
            exclude_reasons.APPEND("near_limit")
        
        quality = "low" IF exclude_flag ELSE "ok"
        
        // 記錄預測結果
        prediction_record = {
            ts: device_group[ts].LAST(),
            device_id: device_id,
            TTW_p10: ttw_quantiles.p10,
            TTW_p50: ttw_quantiles.p50,
            TTW_p90: ttw_quantiles.p90,
            quality: quality
        }
        predictions.APPEND(prediction_record)
    
    // 輸出結果
    prediction_dataframe = CREATE_DATAFRAME(predictions)
    
    SAVE prediction_dataframe TO prediction_file
    
    meta_info = {
        store_id: store_id,
        version: version,
        model_type: "LSTM_trend",
        horizon: horizon,
        exclusion_details: exclude_reasons_per_device
    }
    SAVE meta_info TO meta_file
    
    stats = COMPUTE_STATS(prediction_dataframe)
    
    RETURN prediction_file, meta_file, version, stats

END FUNCTION
```

## 輔助流程

### 3. 資料前處理 (_resample_minutely)

```
FUNCTION _resample_minutely(dataframe):
    
    processed_groups = []
    
    FOR each (store_id, device_id) IN dataframe.GROUPBY([store_id, device_id]):
        group_data = SORT_BY timestamp
        
        // 處理重複時間戳
        IF HAS_DUPLICATE_TIMESTAMPS:
            KEEP_LAST_RECORD_PER_TIMESTAMP
        
        // 重採樣到分鐘級別
        minutely_data = RESAMPLE_TO_1MIN_INTERVALS(group_data)
        FORWARD_FILL missing_values
        
        processed_groups.APPEND(minutely_data)
    
    combined_data = CONCATENATE(processed_groups)
    
    // 計算除霜後時間
    FOR each device_group:
        ingest_ts = COMPUTE_INGEST_TS_FROM_DEFROST_TRANSITIONS(device_group)
        device_group[ingest_ts] = ingest_ts
    
    RETURN combined_data

END FUNCTION
```

### 4. 序列生存資料集 (SeqSurvivalDataset)

```
CLASS SeqSurvivalDataset:
    
    FUNCTION __init__(dataframe, features, device_type_map, window, horizon, step):
        self.samples = []
        
        FOR each device IN dataframe.GROUPBY(device_id):
            device_data = SORT_BY timestamp
            
            // 建立滑動窗口樣本
            FOR i = 0 TO len(device_data) - (window + horizon) STEP step:
                start_index = i
                end_index = i + window + horizon - 1
                self.samples.APPEND((start_index, end_index, device_id))
    
    FUNCTION __getitem__(index):
        start_idx, end_idx, device_id = self.samples[index]
        
        // 提取輸入窗口
        input_window = dataframe[start_idx : start_idx + window]
        future_window = dataframe[start_idx + window : end_idx + 1]
        
        // 特徵處理
        X = input_window[features]
        X = FILL_NA(X, method="forward_fill")
        X = FILL_NA(X, method="backward_fill") 
        X = FILL_NA(X, value=0.0)
        
        // 目標標籤生成
        device_type = device_type_map[device_id]
        safe_temperature = GET_SAFE_TEMP_BY_TYPE(device_type)
        
        future_temps = future_window[temp_current]
        future_temps = FILL_NA(future_temps)
        
        // 生存分析標籤
        y_hazard = ZEROS(horizon)
        mask = ZEROS(horizon)
        
        // 找到第一個越過安全線的時間點
        cross_time = FIND_FIRST_CROSSING_TIME(future_temps, safe_temperature)
        
        IF cross_time EXISTS:
            // 事件發生：在 cross_time 標記事件，之前標記存活
            mask[0:cross_time] = 1
            y_hazard[cross_time-1] = 1
        ELSE:
            // 右截尾：整個期間都標記存活
            mask[0:horizon] = 1
        
        RETURN X, y_hazard, mask, device_id, device_type

END CLASS
```

### 5. 損失函數 (hazard_negative_log_likelihood)

```
FUNCTION hazard_negative_log_likelihood(logits, y_hazard, mask):
    // logits: (batch_size, horizon)
    // y_hazard: (batch_size, horizon) - 事件發生的 one-hot 編碼
    // mask: (batch_size, horizon) - 有效時間區間遮罩
    
    hazard_probs = SIGMOID(logits)  // 每個時間點的風險機率
    
    // 計算 BCE loss
    bce_loss = BINARY_CROSS_ENTROPY(hazard_probs, y_hazard, reduction="none")
    
    // 只計算有效時間區間的損失
    masked_loss = bce_loss * mask
    
    // 正規化損失
    per_sample_loss = SUM(masked_loss, dim=1) / (SUM(mask, dim=1) + epsilon)
    
    RETURN MEAN(per_sample_loss)

END FUNCTION
```