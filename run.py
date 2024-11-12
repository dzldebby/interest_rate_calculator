import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog

def calculate_bank_interest(deposit_amount, bank_tiers):
    """Calculate interest earned for a specific deposit amount in a bank"""
    if deposit_amount <= 0:
        return {
            'total_interest': 0,
            'breakdown': []
        }
        
    total_interest = 0
    remaining_amount = deposit_amount
    breakdown = []
    
    for tier in bank_tiers:
        # Calculate how much money falls into this tier
        amount_in_tier = min(remaining_amount, tier['amount'])
        if amount_in_tier <= 0:
            break
            
        # Calculate interest for this tier
        tier_interest = amount_in_tier * tier['rate']
        monthly_interest = tier_interest / 12
        
        breakdown.append({
            'amount_in_tier': amount_in_tier,
            'tier_rate': tier['rate'],
            'tier_interest': tier_interest,
            'monthly_interest': monthly_interest
        })
        
        total_interest += tier_interest
        remaining_amount -= amount_in_tier
    
    return {
        'total_interest': total_interest,
        'breakdown': breakdown
    }

def create_optimization_problem(banks_data, total_investment):
    """Set up the linear programming problem"""
    num_banks = len(banks_data)
    
    # Prepare coefficients for objective function
    c = []
    bounds = []
    bank_names = []
    
    for bank_name, tiers in banks_data.items():
        bank_names.append(bank_name)
        max_deposit = sum(tier['amount'] for tier in tiers)
        first_tier_rate = -tiers[0]['rate']
        c.append(first_tier_rate)
        bounds.append((0, min(max_deposit, total_investment)))
    
    # Constraint: total investment must equal sum of deposits
    A_eq = [np.ones(num_banks)]
    b_eq = [total_investment]
    
    return {
        'c': np.array(c),
        'A_eq': np.array(A_eq),
        'b_eq': np.array(b_eq),
        'bounds': bounds,
        'bank_names': bank_names
    }

def simple_allocation(banks_data, total_investment):
    """Simple allocation strategy when optimization fails"""
    all_tiers = []
    for bank_name, tiers in banks_data.items():
        for tier in tiers:
            all_tiers.append({
                'bank_name': bank_name,
                'rate': tier['rate'],
                'amount': tier['amount']
            })
    
    all_tiers.sort(key=lambda x: x['rate'], reverse=True)
    remaining_investment = total_investment
    allocations = {}
    total_interest = 0
    
    for tier in all_tiers:
        if remaining_investment <= 0:
            break
            
        bank_name = tier['bank_name']
        amount_for_tier = min(remaining_investment, tier['amount'])
        
        if bank_name not in allocations:
            allocations[bank_name] = {
                'deposit': 0,
                'annual_interest': 0,
                'monthly_interest': 0,
                'breakdown': []
            }
        
        interest = amount_for_tier * tier['rate']
        allocations[bank_name]['deposit'] += amount_for_tier
        allocations[bank_name]['annual_interest'] += interest
        allocations[bank_name]['monthly_interest'] = allocations[bank_name]['annual_interest'] / 12
        allocations[bank_name]['breakdown'].append({
            'amount_in_tier': amount_for_tier,
            'tier_rate': tier['rate'],
            'tier_interest': interest,
            'monthly_interest': interest / 12
        })
        
        total_interest += interest
        remaining_investment -= amount_for_tier
    
    return {
        'allocations': allocations,
        'total_annual_interest': total_interest,
        'total_monthly_interest': total_interest / 12,
        'effective_rate': (total_interest / total_investment) * 100
    }

def optimize_deposits(banks_data, total_investment):
    """Optimize the distribution of deposits across banks"""
    # Convert banks_data to format expected by create_optimization_problem
    simplified_banks_data = {
        bank_name: bank_info['tiers'] 
        for bank_name, bank_info in banks_data.items()
    }
    
    problem = create_optimization_problem(simplified_banks_data, total_investment)
    
    try:
        result = linprog(
            c=problem['c'],
            A_eq=problem['A_eq'],
            b_eq=problem['b_eq'],
            bounds=problem['bounds'],
            method='highs'
        )
        
        if not result.success:
            return simple_allocation(simplified_banks_data, total_investment)
        
        optimization_result = {}
        total_interest = 0
        
        for i, bank_name in enumerate(problem['bank_names']):
            deposit_amount = result.x[i]
            if deposit_amount > 0.01:
                interest_calc = calculate_bank_interest(
                    deposit_amount, 
                    simplified_banks_data[bank_name]
                )
                annual_interest = interest_calc['total_interest']
                monthly_interest = annual_interest / 12
                
                optimization_result[bank_name] = {
                    'deposit': deposit_amount,
                    'annual_interest': annual_interest,
                    'monthly_interest': monthly_interest,
                    'breakdown': interest_calc['breakdown']
                }
                total_interest += annual_interest
        
        return {
            'allocations': optimization_result,
            'total_annual_interest': total_interest,
            'total_monthly_interest': total_interest / 12,
            'effective_rate': (total_interest / total_investment) * 100
        }
        
    except Exception as e:
        return simple_allocation(simplified_banks_data, total_investment)
    

def process_interest_rates(file_path='interest_rates.csv'):
    """Process interest rates from CSV file"""
    df = pd.read_csv(file_path)
    banks_data = {}
    
    for _, row in df.iterrows():
        bank_name = row['bank']
        requires_salary = row['credit_salary'] == 'Y'
        # Clean up the remarks text and combine with others
        remarks = str(row['remarks']) if pd.notna(row['remarks']) else ""
        others = str(row['others']) if pd.notna(row['others']) else ""
        
        # Combine requirements, clean up formatting
        other_requirements = []
        if others:
            other_requirements.append(others)
        if remarks:
            other_requirements.append(remarks)
            
        tiers = []
        
        for i in range(1, 7):
            rate_col = f'interest_rate_{i}'
            amount_col = f'amount_{i}'
            
            if rate_col not in df.columns or amount_col not in df.columns:
                continue
            
            if pd.isna(row[rate_col]) and pd.isna(row[amount_col]):
                continue
                
            rate = float(row[rate_col].strip('%')) / 100 if pd.notna(row[rate_col]) else None
            amount = float(row[amount_col]) if pd.notna(row[amount_col]) else None
            
            if rate is not None and amount is not None:
                tiers.append({
                    'rate': rate,
                    'amount': amount
                })
        
        banks_data[bank_name] = {
            'tiers': tiers,
            'requires_salary': requires_salary,
            'other_requirements': other_requirements  # Now it's a list of requirements
        }
    
    return banks_data


def optimize_with_salary_constraint(banks_data, total_investment):
    """Two-phase optimization considering salary crediting constraint"""
    
    # Phase 1: Initial optimization
    initial_results = optimize_deposits(banks_data, total_investment)
    
    # Get banks that require salary from the optimal allocation
    salary_requiring_banks = [
        bank_name for bank_name in initial_results['allocations'].keys()
        if banks_data[bank_name]['requires_salary']
    ]
    
    # If no salary-requiring banks in solution, we're done
    if len(salary_requiring_banks) == 0:
        initial_results['chosen_salary_bank'] = None
        return initial_results
        
    # If only one salary-requiring bank, it's automatically chosen
    if len(salary_requiring_banks) == 1:
        initial_results['chosen_salary_bank'] = salary_requiring_banks[0]
        return initial_results
    
    # Phase 2: Try each salary-requiring bank as the chosen one
    best_result = None
    best_interest = 0
    
    for chosen_bank in salary_requiring_banks:
        # Create modified banks_data with zero interest for non-chosen salary banks
        modified_banks_data = {}
        for bank_name, bank_info in banks_data.items():
            if bank_info['requires_salary'] and bank_name != chosen_bank:
                # Create version with zero interest rates
                modified_banks_data[bank_name] = {
                    'tiers': [{'rate': 0.0, 'amount': tier['amount']} 
                             for tier in bank_info['tiers']],
                    'requires_salary': True
                }
            else:
                modified_banks_data[bank_name] = bank_info
        
        # Run optimization with modified data
        result = optimize_deposits(modified_banks_data, total_investment)
        
        # Keep track of best result
        if result['total_annual_interest'] > best_interest:
            best_interest = result['total_annual_interest']
            best_result = result
            best_result['chosen_salary_bank'] = chosen_bank
    
    return best_result


def streamlit_app():
    st.title("Bank Interest Rate Optimizer")
    
    # Add password input at the top
    password = st.text_input("Enter password:", type="password")
    
    # Check if password is correct
    if password != "interest":
        st.error("Please enter the correct password to continue.")
        return  # Stop here if password is incorrect
    
    try:
        # Load and process the CSV file directly
        banks_data = process_interest_rates('interest_rates.csv')
        
        # Investment amount input
        st.subheader("Enter Investment Amount")
        investment_amount = st.number_input(
            "Investment amount ($)", 
            min_value=0.0, 
            value=100000.0, 
            step=1000.0,
            format="%0.2f"
        )
        
        if st.button("Optimize Investment"):
            # Calculate optimal distribution with salary constraint
            results = optimize_with_salary_constraint(banks_data, investment_amount)

            # Display Summary First with Monthly Interest Highlighted
            st.subheader("Investment Summary")
            st.markdown(f"### Monthly Interest: :green[${results['total_monthly_interest']:,.2f}]")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Total Annual Interest: ${results['total_annual_interest']:,.2f}")
            with col2:
                st.write(f"Effective Interest Rate: {results['effective_rate']:.2f}%")
            
            # Display Optimal Distribution in a Table
            st.subheader("Optimal Distribution")
            
            # First display the allocation table
            distribution_data = []
            for bank_name, data in results['allocations'].items():
                distribution_data.append({
                    'Bank': bank_name,
                    'Deposit Amount': f"${data['deposit']:,.2f}",
                    'Annual Interest': f"${data['annual_interest']:,.2f}",
                    'Monthly Interest': f"${data['monthly_interest']:,.2f}",
                    'Requires Salary Credit': 'Yes' if banks_data[bank_name]['requires_salary'] else 'No'
                })
            
            st.table(pd.DataFrame(distribution_data))

            
            # Display if there's a chosen salary bank
            if 'chosen_salary_bank' in results:
                st.info(f"💰 Salary must be credited to: {results['chosen_salary_bank']}")
            

            # Add a new section for requirements
            st.subheader("Bank Requirements")
            for bank_name, data in results['allocations'].items():
                has_requirements = False
                st.markdown(f"**{bank_name}**")
                
                if banks_data[bank_name]['requires_salary']:
                    has_requirements = True
                    st.markdown("- Credit salary to this account")
                
                # Display other requirements as separate bullet points
                for req in banks_data[bank_name]['other_requirements']:
                    if req.strip():  # Only display non-empty requirements
                        has_requirements = True
                        # Replace $ with \$ to escape the LaTeX interpretation
                        formatted_req = req.strip().replace("$", "\\$")
                        st.markdown(f"- {formatted_req}")
                
                if has_requirements:
                    st.markdown("")  # Add spacing between banks
                else:
                    st.markdown("- No special requirements")
                    st.markdown("")
            
            # Detailed Breakdown (collapsible)
            with st.expander("See Detailed Breakdown"):
                for bank_name, data in results['allocations'].items():
                    requires_salary = banks_data[bank_name]['requires_salary']
                    salary_text = " (Requires Salary)" if requires_salary else ""
                    st.write(f"\n**{bank_name}{salary_text}:**")
                    tier_data = []
                    for tier in data['breakdown']:
                        tier_data.append({
                            'Amount in Tier': f"${tier['amount_in_tier']:,.2f}",
                            'Interest Rate': f"{tier['tier_rate']*100:.2f}%",
                            'Annual Interest': f"${tier['tier_interest']:,.2f}",
                            'Monthly Interest': f"${tier['monthly_interest']:,.2f}"
                        })
                    st.table(pd.DataFrame(tier_data))
            
            # Add visual separation before Comparison Analysis
            st.markdown("")  # Adds blank line
            st.markdown("---")  # Adds horizontal rule
            st.markdown("")  # Adds another blank line for spacing
            
            # Comparison Analysis
            st.subheader("Comparison with Other Scenarios")
            with st.expander("See Comparison Analysis"):
                # Equal Distribution Scenario
                equal_amount = investment_amount / len(banks_data)
                total_interest_equal = 0
                
                # Assuming one salary bank for equal distribution
                salary_banks = [bank for bank, info in banks_data.items() if info['requires_salary']]
                chosen_salary_bank = salary_banks[0] if salary_banks else None
                
                for bank_name, bank_info in banks_data.items():
                    # If bank requires salary but isn't chosen, use zero interest
                    if bank_info['requires_salary'] and bank_name != chosen_salary_bank:
                        total_interest_equal += 0
                    else:
                        interest_calc = calculate_bank_interest(equal_amount, bank_info['tiers'])
                        total_interest_equal += interest_calc['total_interest']
                
                equal_monthly = total_interest_equal / 12
                monthly_difference_equal = results['total_monthly_interest'] - equal_monthly
                
                # Single Bank Scenarios
                comparison_data = []
                
                for bank_name, bank_info in banks_data.items():
                    interest_calc = calculate_bank_interest(investment_amount, bank_info['tiers'])
                    annual_interest = interest_calc['total_interest']
                    monthly_interest = annual_interest / 12
                    monthly_difference = results['total_monthly_interest'] - monthly_interest
                    
                    comparison_data.append({
                        'Bank': bank_name,
                        'Monthly Interest': f"${monthly_interest:,.2f}",
                        'Difference vs Optimal': f"${monthly_difference:,.2f} less",
                        'Effective Rate': f"{(annual_interest/investment_amount)*100:.2f}%",
                        'Requires Salary Credit': 'Yes' if bank_info['requires_salary'] else 'No'
                    })
                
                # Display comparison table
                st.table(pd.DataFrame(comparison_data))
                
                # Display optimization advantage
                st.markdown("### Optimization Advantage")
                st.markdown(f"By using the optimal distribution instead of equal distribution:")
                st.markdown(f"- Extra monthly earnings: :green[${monthly_difference_equal:,.2f}]")
                st.markdown(f"- Extra annual earnings: :green[${monthly_difference_equal*12:,.2f}]")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    streamlit_app()