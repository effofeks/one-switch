SELECT 
    carton_groupings.id, 
    carton_groupings.line_item_id,
    carton_groupings.order_id,
    carton_groupings.state,
    carton_groupings.pallet_number,
    carton_groupings.sequence_number,
    carton_groupings.exporter_code,
    carton_groupings.farm_code,
    carton_groupings.packhouse_code,
    production_regions.name production_region,
    commodities.name commodity_name,
    varieties.name variety_name,
    carton_groupings.cartons,
    carton_groupings.pallet_stack,
    carton_groupings.pack,
    carton_groupings.size_count,
    carton_groupings.grade,
    carton_groupings.orchard,
    carton_groupings.net_mass,
    carton_groupings.container_number,
    carton_groupings.mark,
    carton_groupings.map_local_market local_market,
    carton_groupings.map_jbin jbin,
    carton_groupings.target_market,
    carton_groupings.target_region,
    carton_groupings.target_country,
    carton_groupings.pallet_gross_mass,
    carton_groupings.seller_id,
    cg_sellers.types seller_types,
    carton_groupings.buyer_id,
    cg_buyers.types buyer_types,
    carton_groupings.packing_week,
    carton_groupings.batch_number,
    carton_groupings.inventory_code,
    carton_groupings.consignment_number,
    carton_groupings.pallet_rejected,
    carton_groupings.commercial_term_id,
    carton_groupings.advance_price,
    carton_groupings.advance_due_date,
    carton_groupings.final_price,
    carton_groupings.final_due_date,
    carton_groupings.currency,
    carton_groupings.transport_type,
    carton_groupings.packed_datetime,
    carton_groupings.first_event_datetime

FROM 
    carton_groupings

    LEFT JOIN 
        commodities 
        ON carton_groupings.commodity_id = commodities.id
        
    LEFT JOIN 
        varieties 
        ON carton_groupings.variety_id = varieties.id
        
    LEFT JOIN 
        food_business_operators
        ON carton_groupings.farm_code = food_business_operators.fbo_code
        
    LEFT JOIN 
        production_regions 
        ON food_business_operators.production_region_id = production_regions.id
        
    LEFT JOIN 
        cg_owners cg_buyers
        ON carton_groupings.id = cg_buyers.carton_grouping_id AND carton_groupings.buyer_id = cg_buyers.company_id
    
    LEFT JOIN 
        cg_owners cg_sellers
        ON carton_groupings.id = cg_sellers.carton_grouping_id AND carton_groupings.seller_id = cg_sellers.company_id

WHERE 
    map_season_year = 2024 AND commodities.commodity_group = 'Citrus' 