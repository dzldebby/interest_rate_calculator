BANK INTEREST RATE OPTIMIZER - TECHNICAL DOCUMENTATION
===================================================

[Previous sections remain the same until 2.2]

2.2 Interest Rate Processing
---------------------------
Function: process_interest_rates(file_path='interest_rates.csv')

Detailed Implementation:
1. Reads CSV using pandas.read_csv()
2. Iterates through each row using df.iterrows()
3. For each bank:
   - Extracts bank name from 'bank' column
   - Processes up to 6 tiers using range(1, 7)
   - For each tier:
     * Constructs column names (interest_rate_{i}, amount_{i})
     * Validates data presence using pd.isna()
     * Converts interest rate from percentage string to decimal (e.g., "3.00%" → 0.03)
     * Validates both rate and amount are present
     * Creates tier dictionary with {'rate': rate, 'amount': amount}
4. Returns structured dictionary:
   {
       'BankA': [
           {'rate': 0.03, 'amount': 75000},
           {'rate': 0.045, 'amount': 50000},
           ...
       ],
       'BankB': [...]
   }

2.3 Interest Calculation
-----------------------
Function: calculate_bank_interest(deposit_amount, bank_tiers)

Detailed Implementation:
1. Input validation:
   - Checks if deposit_amount > 0
   - Returns zero values if invalid
2. Tier processing:
   - Maintains running total of interest and remaining amount
   - For each tier:
     * Calculates amount that falls into current tier using min()
     * Breaks if no remaining amount
     * Calculates tier interest: amount_in_tier * tier_rate
     * Calculates monthly interest: tier_interest / 12
     * Creates detailed breakdown entry
     * Updates total interest and remaining amount
3. Returns structured result:
   {
       'total_interest': float,
       'breakdown': [
           {
               'amount_in_tier': float,
               'tier_rate': float,
               'tier_interest': float,
               'monthly_interest': float
           },
           ...
       ]
   }

2.4 Optimization Engine
----------------------
a. Problem Setup
Function: create_optimization_problem(banks_data, total_investment)

Detailed Implementation:
1. Problem structure creation:
   - Counts number of banks for matrix dimensions
   - For each bank:
     * Calculates maximum possible deposit (sum of tier amounts)
     * Gets first tier rate for initial optimization
     * Creates bounds tuple (0, min(max_deposit, total_investment))
     * Stores bank name for result mapping
2. Constraint setup:
   - Creates equality constraint matrix (A_eq) ensuring sum equals total investment
   - Creates bounds list ensuring non-negative deposits within bank limits
3. Returns optimization parameters:
   {
       'c': numpy array of negative first tier rates,
       'A_eq': numpy array for constraints,
       'b_eq': numpy array [total_investment],
       'bounds': list of (min, max) tuples,
       'bank_names': list of bank names
   }

b. Optimization Execution
Function: optimize_deposits(banks_data, total_investment)

Detailed Implementation:
1. Gets optimization problem structure
2. Executes scipy.optimize.linprog:
   - Uses 'highs' method for efficiency
   - Minimizes negative interest (maximizes positive interest)
   - Applies constraints from problem setup
3. Success handling:
   - Checks result.success
   - If successful:
     * Processes each bank's optimal deposit
     * Calculates detailed interest breakdown
     * Computes total interest and rates
   - If fails:
     * Falls back to simple_allocation()
4. Returns comprehensive result:
   {
       'allocations': {
           'BankA': {
               'deposit': float,
               'annual_interest': float,
               'monthly_interest': float,
               'breakdown': [tier_details]
           },
           ...
       },
       'total_annual_interest': float,
       'total_monthly_interest': float,
       'effective_rate': float
   }

c. Fallback Strategy
Function: simple_allocation(banks_data, total_investment)

Detailed Implementation:
1. Tier collection:
   - Flattens all tiers from all banks into single list
   - Adds bank name to each tier for tracking
2. Optimization process:
   - Sorts tiers by interest rate (descending)
   - Allocates money to highest rates first
   - Tracks remaining investment amount
3. Result construction:
   - Creates same structure as optimize_deposits()
   - Ensures consistent return format for seamless fallback

[Previous sections remain the same...]

2.5 User Interface (Streamlit)
-----------------------------
Function: streamlit_app()

Detailed Implementation:
1. Application Setup:
   - Sets title using st.title()
   - Initializes error handling with try-except block
   - Loads bank data using process_interest_rates()

2. Input Handling:
   - Creates investment amount input field:
     * Uses st.number_input()
     * Sets constraints:
       - min_value=0.0
       - step=1000.0
       - format="%0.2f"
     * Default value: 100000.0
   - Creates optimization trigger button using st.button()

3. Results Display Structure:
   a. Investment Summary Section:
      - Displays prominent monthly interest in green
      - Uses two-column layout (st.columns):
        * Column 1: Annual Interest
        * Column 2: Effective Interest Rate
      - Formats all monetary values with commas and 2 decimal places

   b. Optimal Distribution Table:
      - Creates structured DataFrame for display:
        * Columns: Bank, Deposit Amount, Annual Interest, Monthly Interest
        * Formats all monetary values with "$" prefix
      - Uses st.table() for fixed layout display

   c. Detailed Breakdown (Expandable):
      - Creates collapsible section using st.expander()
      - For each bank:
        * Shows bank name as header
        * Creates tier-by-tier breakdown table
        * Displays:
          - Amount in each tier
          - Interest rate percentage
          - Annual and monthly interest
        * Uses st.table() for consistent formatting

2.6 Comparison Analysis
----------------------
Implemented within streamlit_app()

Detailed Implementation:
1. Equal Distribution Analysis:
   a. Calculation:
      - Divides total investment by number of banks
      - For each bank:
        * Calculates interest using calculate_bank_interest()
        * Accumulates total interest
      - Computes monthly figures (total/12)
      - Calculates difference from optimal solution

2. Single Bank Scenarios:
   a. Data Collection:
      - Iterates through each bank
      - Calculates results if all money was in one bank:
        * Uses calculate_bank_interest() with full amount
        * Computes annual and monthly interest
        * Calculates difference from optimal solution
        * Determines effective interest rate

   b. Results Compilation:
      - Creates comparison DataFrame with columns:
        * Bank name
        * Monthly Interest (formatted with $)
        * Difference vs Optimal (formatted with $ and "less")
        * Effective Rate (formatted with %)
      - Uses st.table() for display

3. Optimization Advantage Display:
   a. Formatting:
      - Uses markdown for enhanced formatting
      - Highlights key figures in green
   b. Shows:
      - Extra monthly earnings vs equal distribution
      - Extra annual earnings (monthly * 12)

4. Error Handling in UI:
   a. CSV Loading Errors:
      - Catches file not found
      - Handles invalid CSV format
      - Displays user-friendly error messages
   
   b. Calculation Errors:
      - Catches division by zero
      - Handles invalid input values
      - Shows specific error messages

   c. Display Errors:
      - Handles missing data gracefully
      - Maintains UI stability on error
      - Provides clear error feedback

5. UI Performance Optimizations:
   a. Data Loading:
      - Loads CSV once at startup
      - Caches bank data for reuse
   
   b. Calculations:
      - Performs heavy calculations only on button click
      - Uses efficient data structures for display
   
   c. Display Updates:
      - Updates only changed values
      - Uses efficient table formatting
      - Minimizes UI redraws

6. User Experience Features:
   a. Input Validation:
      - Prevents negative values
      - Enforces reasonable step sizes
      - Provides immediate feedback
   
   b. Results Clarity:
      - Hierarchical information display
      - Clear visual hierarchy
      - Consistent formatting
   
   c. Interactive Elements