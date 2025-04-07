SELECT 
    carton_groupings.id,
    carton_groupings.pallet_number,
    carton_groupings.container_number,
    carton_groupings.seller_id,
    carton_groupings.buyer_id,
    carton_groupings.packing_week,
    production_regions.name production_region,
    carton_groupings.orchard,
    commodities.name commodity_name,
    varieties.name variety_name,
    carton_groupings.map_size_count size_count,
    CASE 
        WHEN carton_groupings.size_count ~ '^\d+(\.0+)?$' and commodities.name <> 'Soft Citrus' THEN
        CASE
            -- Lemon sizing
            WHEN LOWER(commodities.name) = 'lemon' THEN
                CASE
                    WHEN carton_groupings.size_count::numeric::int >= 189 THEN 'Extra Small'
                    WHEN carton_groupings.size_count::numeric::int >= 138 THEN 'Small'
                    WHEN carton_groupings.size_count::numeric::int >= 100 THEN 'Medium'
                    WHEN carton_groupings.size_count::numeric::int >= 75 THEN 'Large'
                    ELSE 'Large'
                END

            -- Valencia oranges and similar
            WHEN LOWER(commodities.name) = 'orange' THEN
                CASE
                    WHEN carton_groupings.size_count::numeric::int <= 48 THEN 'Large'
                    WHEN carton_groupings.size_count::numeric::int <= 64 THEN 'Medium'
                    WHEN carton_groupings.size_count::numeric::int <= 88 THEN 'Small'
                    WHEN carton_groupings.size_count::numeric::int <= 125 THEN 'Extra Small'
                    ELSE 'Mixed/Non-standard'
                END

            -- Star Ruby grapefruit sizing
            WHEN LOWER(commodities.name) = 'grapefruit' THEN
                CASE
                    WHEN carton_groupings.size_count::numeric::int <= 35 THEN 'Large'
                    WHEN carton_groupings.size_count::numeric::int <= 45 THEN 'Medium'
                    WHEN carton_groupings.size_count::numeric::int <= 55 THEN 'Small'
                    WHEN carton_groupings.size_count::numeric::int = 60 THEN 'Extra Small'
                    ELSE 'Mixed/Non-standard'
                END

            -- Default for numeric but uncategorized commodity
            ELSE 'Unknown'
        END

    -- Non-numeric soft citrus codes and overrides
    ELSE
        CASE
            WHEN LOWER(commodities.name) = 'soft citrus' THEN
                CASE
                    WHEN carton_groupings.size_count ILIKE ANY (ARRAY['1XXX', '1XX']) THEN 'Large'
                    WHEN carton_groupings.size_count ILIKE ANY (ARRAY['1X', '1', '1.0']) THEN 'Medium'
                    WHEN carton_groupings.size_count IN ('2', '3', '2.0', '3.0') THEN 'Small'
                    WHEN carton_groupings.size_count IN ('4', '5', '6', '4.0', '5.0', '6.0') THEN 'Extra Small'
                    ELSE 'Mixed/Non-standard'
                END

            -- Literal character groupings and fallback mappings
            WHEN carton_groupings.size_count ILIKE 'M' THEN 'Medium'
            WHEN carton_groupings.size_count ILIKE 'L' THEN 'Large'
            WHEN carton_groupings.size_count ILIKE 'XL' THEN 'Large'
            WHEN carton_groupings.size_count ILIKE 'XS' THEN 'Extra Small'
			WHEN carton_groupings.size_count ILIKE 'S' THEN 'Small'
            -- Known invalid/mixed inputs
            WHEN carton_groupings.size_count ~* 'mix|juice|juic|var|waste|xx|lm|ms|mx|/' THEN 'Mixed/Non-standard'
            ELSE 'Unknown'
        END
    END AS size_categorization,
    carton_groupings.map_class class,
    carton_groupings.pallet_stack,
    carton_groupings.pack,
    carton_groupings.map_local_market local_market,
    carton_groupings.map_jbin jbin,
    carton_groupings.target_market,
    carton_groupings.target_region,
    carton_groupings.target_country,
    carton_groupings.cartons,
    mv_pallet_timeline."STD Cartons" std_cartons


FROM 
    carton_groupings

    LEFT JOIN 
        data_science.mv_pallet_timeline
        ON carton_groupings.id = mv_pallet_timeline.cg_id

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

WHERE 
    carton_groupings.map_season_year = 2024 AND mv_pallet_timeline.commodity_group = 'Citrus' 
