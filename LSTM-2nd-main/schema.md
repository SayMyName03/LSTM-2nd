# Database Schema Documentation

## Entity-Relationship Diagram

The stock prediction application uses a SQLite database with the following entity relationships:

```
+-------------------+      +-------------------+      +-------------------+
| StockPrediction   |      | UserPreference    |      | AnalysisHistory   |
+-------------------+      +-------------------+      +-------------------+
| id (PK)           |      | id (PK)           |      | id (PK)           |
| ticker            |      | user_id           |      | ticker            |
| prediction_date   |      | ticker            |      | analysis_date     |
| current_price     |      | added_date        |      | successful        |
| predicted_prices  |      |                   |      | error_message     |
| prediction_dates  |      |                   |      |                   |
+-------------------+      +-------------------+      +-------------------+
```

## Tables Description

### StockPrediction
Stores the results of stock prediction analyses.

| Column           | Type      | Description                                       |
|------------------|-----------|---------------------------------------------------|
| id               | Integer   | Primary key                                       |
| ticker           | String    | Stock symbol (e.g., AAPL, GOOGL)                  |
| prediction_date  | DateTime  | When the prediction was generated                 |
| current_price    | Float     | Stock price at time of prediction                 |
| predicted_prices | Text      | JSON string of predicted future prices            |
| prediction_dates | Text      | JSON string of dates for predictions              |

### UserPreference
Stores user's favorite or frequently analyzed stocks.

| Column     | Type      | Description                                       |
|------------|-----------|---------------------------------------------------|
| id         | Integer   | Primary key                                       |
| user_id    | String    | User identifier (simplified in this implementation)|
| ticker     | String    | Stock symbol (e.g., AAPL, GOOGL)                  |
| added_date | DateTime  | When the stock was added to preferences           |

### AnalysisHistory
Logs all stock analysis requests, including errors.

| Column        | Type      | Description                                       |
|---------------|-----------|---------------------------------------------------|
| id            | Integer   | Primary key                                       |
| ticker        | String    | Stock symbol that was analyzed                    |
| analysis_date | DateTime  | When the analysis was performed                   |
| successful    | Boolean   | Whether the analysis completed successfully       |
| error_message | Text      | Error message if analysis failed (null otherwise) |

## Data Flow

1. When a user requests a stock prediction:
   - An entry is created in AnalysisHistory
   - If prediction is successful, the results are stored in StockPrediction

2. Users can save preferred stocks to UserPreference for quick access

3. Historical predictions can be retrieved for comparison and analysis

## Relationships

- There is no direct foreign key relationship between these tables
- They are linked conceptually through the ticker symbol
- This design allows for flexibility and simplicity in data management
