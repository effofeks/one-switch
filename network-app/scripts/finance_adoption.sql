WITH cte_data AS (
    SELECT 
        carton_groupings.packing_week,
        commodities.commodity_group,
        COUNT(DISTINCT carton_groupings.id) "total_cgs",
        COUNT(DISTINCT t1.cg_id) "finance_cgs"
    FROM 
        carton_groupings 
        LEFT JOIN 
            (SELECT DISTINCT ON (cg_id) cg_id FROM data_science.dso_finance ) t1
            ON carton_groupings.id = t1.cg_id
            
        LEFT JOIN 
            commodities 
            ON carton_groupings.commodity_id = commodities.id
            
    WHERE 
        carton_groupings.map_season_year in (2024, 2025) AND carton_groupings.packing_week IS NOT NULL
    
            
    GROUP BY 
        1, 2 )
        
SELECT
    *, 
	finance_cgs/NULLIF(total_cgs, 0)::FLOAT prop_finance_cgs
    
FROM 
    cte_data 