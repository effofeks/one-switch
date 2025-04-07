SELECT 
    dso_finance.invoice_line_item_id,
    dso_finance.cg_id,
    dso_finance.company_id,
    dso_finance.document_type,
    dso_finance.invoice_id,
    dso_finance.cost_type,
    dso_finance.ili_price_minor_unit,
    dso_finance.currency,
    dso_finance.price_unit,
    dso_finance.cgt_amount,
    dso_finance.stuff_date,
    dso_finance.is_actual,
    dso_finance.final_due_date,
    dso_finance.advance_due_date,
    dso_finance.order_price,
    dso_finance.order_price_per_carton,
    dso_finance.order_price_unit,
    dso_finance.order_currency,
    dso_finance.ct_advance_amount_per_carton,
    dso_finance.ct_advance_currency,
    dso_finance.ct_advance_week,
    dso_finance.ct_final_value,
    dso_finance.ct_final_currency,
    dso_finance.ct_advance_credit_term,
    dso_finance.ct_final_credit_term,
    dso_finance.incoterm,
    dso_finance.actual_advance_currency,
    dso_finance.advances_exchange_rate,
    dso_finance.actual_advance,
    dso_finance.advance_transaction_week,
    dso_finance.actual_advance_zar,
    dso_finance.actual_final_currency,
    dso_finance.finals_exchange_rate,
    dso_finance.actual_final,
    dso_finance.actual_final_zar,
    dso_finance.actual_final_per_pallet,
    dso_finance.actual_final_per_carton,
    dso_finance.total_value_per_pallet,
    dso_finance.final_transaction_week,
    dso_finance.total_value_per_carton,
    dso_finance.exchange_rate,
    dso_finance.payment_status,
    dso_finance.invoice_date,
    dso_finance.account_type

FROM 
    data_science.dso_finance

    LEFT JOIN 
        carton_groupings 
        ON dso_finance.cg_id = carton_groupings.id

    INNER JOIN 
        commodities 
        ON carton_groupings.commodity_id = commodities.id 

WHERE 
    carton_groupings.map_season_year = 2024 AND commodities.commodity_group = 'Citrus'