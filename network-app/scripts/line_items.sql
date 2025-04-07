SELECT 
    DISTINCT ON (line_items.id)
    line_items.id, 
    line_items.quantity,
    line_items.quantity_unit,
    line_items.price_minor_unit,
    line_items.price_unit,
    line_items.currency,
    line_items.pack,
    line_items.price_term,
    line_items.additional_fields,
    line_items.pallet_stack,
    line_items.unlimited,
    line_items.target_market,
    line_items.target_region,
    line_items.target_country,
    line_items.packing_week,
    line_items.incoterm,
    line_items.deleted,
    line_items.grade,
    line_items.state,
    line_items.line_item_grouping_id,
    line_items.rank,
    line_items.batch_number,
    line_items.inventory_code,
    line_items.planned_quantity,
    line_items.planned_quantity_unit,
    line_items.size_counts,
    line_item_groupings.cumulative_quantity,
    line_item_groupings.quantity_type,
    line_item_groupings.common_fields

FROM 
    line_items 
    LEFT JOIN 
    
    carton_groupings
    ON line_items.id = carton_groupings.line_item_id
    
    LEFT JOIN 
        commodities 
        ON carton_groupings.commodity_id = commodities.id

    LEFT JOIN 
        line_item_groupings
        ON line_items.line_item_grouping_id = line_item_groupings.id 

WHERE 
    map_season_year = 2024 AND commodities.commodity_group = 'Citrus' 