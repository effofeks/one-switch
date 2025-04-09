SELECT 
    dso_finance.cg_id,
    dso_finance.document_type,
    dso_finance.cost_type,
    dso_finance.currency,
    dso_finance.cgt_amount,
    dso_finance.exchange_rate,
    dso_finance.payment_status,
    dso_finance.price_unit,
    dso_finance.incoterm

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