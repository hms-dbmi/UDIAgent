"""
Auto-generated visualization tool definitions.

Generated from: src/skills/template_visualizations.json
Schema: data/data_domains/hubmap_data_schema.json
Tools: 52

DO NOT EDIT — regenerate with: python src/generate_tools.py
"""


# Schema metadata (entity URLs and relationships)
SCHEMA = {'entities': {'datasets': {'fields': {'ablation_distance_between_shots_x_units': 'nominal',
                                      'ablation_distance_between_shots_x_value': 'quantitative',
                                      'ablation_distance_between_shots_y_units': 'nominal',
                                      'ablation_distance_between_shots_y_value': 'quantitative',
                                      'ablation_frequency_unit': 'nominal',
                                      'ablation_frequency_value': 'quantitative',
                                      'acquisition_id': 'nominal',
                                      'acquisition_instrument_model': 'nominal',
                                      'acquisition_instrument_vendor': 'nominal',
                                      'amount_of_input_analyte_unit': 'nominal',
                                      'amount_of_input_analyte_value': 'quantitative',
                                      'analyte_class': 'nominal',
                                      'area_normalized_ion_dose_unit': 'nominal',
                                      'area_normalized_ion_dose_value': 'quantitative',
                                      'assay_category': 'nominal',
                                      'assay_input_entity': 'nominal',
                                      'assay_type': 'nominal',
                                      'barcode_offset': 'nominal',
                                      'barcode_read': 'nominal',
                                      'barcode_size': 'nominal',
                                      'bead_barcode_offset': 'nominal',
                                      'bead_barcode_read': 'nominal',
                                      'bead_barcode_size': 'nominal',
                                      'bulk_atac_cell_isolation_protocols_io_doi': 'nominal',
                                      'bulk_rna_isolation_protocols_io_doi': 'nominal',
                                      'bulk_rna_isolation_quality_metric_value': 'quantitative',
                                      'bulk_rna_yield_units_per_tissue_unit': 'nominal',
                                      'bulk_rna_yield_value': 'quantitative',
                                      'bulk_transposition_input_number_nuclei': 'quantitative',
                                      'capture_area_id': 'nominal',
                                      'capture_batch_id': 'nominal',
                                      'cell_barcode_offset': 'nominal',
                                      'cell_barcode_read': 'nominal',
                                      'cell_barcode_size': 'nominal',
                                      'cell_boundary_marker_or_stain': 'nominal',
                                      'created_by_user_displayname': 'nominal',
                                      'created_by_user_email': 'nominal',
                                      'created_timestamp': 'quantitative',
                                      'data_collection_mode': 'nominal',
                                      'data_precision_bytes': 'quantitative',
                                      'dataset_type': 'nominal',
                                      'description': 'nominal',
                                      'desi_solvent': 'nominal',
                                      'desi_solvent_flow_rate': 'quantitative',
                                      'desi_solvent_flow_rate_unit': 'nominal',
                                      'dms': 'nominal',
                                      'dna_assay_input_unit': 'nominal',
                                      'dna_assay_input_value': 'quantitative',
                                      'donor.hubmap_id': 'nominal',
                                      'dual_count_start': 'quantitative',
                                      'end_datetime': 'nominal',
                                      'execution_datetime': 'nominal',
                                      'expected_cell_count': 'quantitative',
                                      'expected_entity_capture_count': 'quantitative',
                                      'gdna_fragmentation_quality_assurance': 'nominal',
                                      'group_name': 'nominal',
                                      'hubmap_id': 'nominal',
                                      'increment_z_unit': 'nominal',
                                      'increment_z_value': 'quantitative',
                                      'intended_tile_overlap_percentage': 'quantitative',
                                      'ion_mobility': 'nominal',
                                      'is_batch_staining_done': 'nominal',
                                      'is_image_preprocessing_required': 'nominal',
                                      'is_staining_automated': 'nominal',
                                      'is_targeted': 'nominal',
                                      'is_technical_replicate': 'nominal',
                                      'labeling': 'nominal',
                                      'last_modified_timestamp': 'quantitative',
                                      'lc_column_model': 'nominal',
                                      'lc_column_vendor': 'nominal',
                                      'lc_flow_rate_unit': 'nominal',
                                      'lc_flow_rate_value': 'quantitative',
                                      'lc_gradient': 'nominal',
                                      'lc_id_unit': 'nominal',
                                      'lc_id_value': 'quantitative',
                                      'lc_instrument_model': 'nominal',
                                      'lc_instrument_vendor': 'nominal',
                                      'lc_length_unit': 'nominal',
                                      'lc_length_value': 'quantitative',
                                      'lc_mobile_phase_a': 'nominal',
                                      'lc_mobile_phase_b': 'nominal',
                                      'lc_resin': 'nominal',
                                      'lc_temp_unit': 'nominal',
                                      'lc_temp_value': 'quantitative',
                                      'library_adapter_sequence': 'nominal',
                                      'library_average_fragment_size': 'quantitative',
                                      'library_concentration_unit': 'nominal',
                                      'library_concentration_value': 'quantitative',
                                      'library_construction_method': 'nominal',
                                      'library_construction_protocols_io_doi': 'nominal',
                                      'library_creation_date': 'nominal',
                                      'library_final_yield': 'quantitative',
                                      'library_final_yield_unit': 'nominal',
                                      'library_final_yield_value': 'quantitative',
                                      'library_id': 'nominal',
                                      'library_input_amount_unit': 'nominal',
                                      'library_input_amount_value': 'quantitative',
                                      'library_layout': 'nominal',
                                      'library_output_amount_unit': 'nominal',
                                      'library_output_amount_value': 'quantitative',
                                      'library_pcr_cycles': 'quantitative',
                                      'library_pcr_cycles_for_sample_index': 'quantitative',
                                      'library_preparation_kit': 'nominal',
                                      'mapped_area_unit': 'nominal',
                                      'mapped_area_value': 'quantitative',
                                      'mapped_consortium': 'nominal',
                                      'mass_resolving_power': 'quantitative',
                                      'max_x_width_unit': 'nominal',
                                      'max_x_width_value': 'quantitative',
                                      'max_y_height_unit': 'nominal',
                                      'max_y_height_value': 'quantitative',
                                      'metadata_schema_id': 'nominal',
                                      'ms_scan_mode': 'nominal',
                                      'ms_source': 'nominal',
                                      'mz_range_high_value': 'quantitative',
                                      'mz_range_low_value': 'quantitative',
                                      'mz_resolving_power': 'quantitative',
                                      'nuclear_marker_or_stain': 'nominal',
                                      'number_of_antibodies': 'quantitative',
                                      'number_of_barcode_probes': 'quantitative',
                                      'number_of_barcode_regions_per_barcode_probe': 'quantitative',
                                      'number_of_biomarker_imaging_rounds': 'quantitative',
                                      'number_of_channels': 'quantitative',
                                      'number_of_cycles': 'quantitative',
                                      'number_of_imaging_rounds': 'quantitative',
                                      'number_of_input_cells_or_nuclei': 'quantitative',
                                      'number_of_iterations_of_cdna_amplification': 'quantitative',
                                      'number_of_pcr_cycles_for_indexing': 'quantitative',
                                      'number_of_pre_amplification_pcr_cycles': 'quantitative',
                                      'number_of_pseudocolors_per_channel': 'quantitative',
                                      'number_of_readout_probes_per_channel': 'quantitative',
                                      'number_of_sections': 'quantitative',
                                      'number_of_spots': 'quantitative',
                                      'number_of_total_imaging_rounds': 'quantitative',
                                      'operator': 'nominal',
                                      'operator_email': 'nominal',
                                      'origin_samples_unique_mapped_organs': 'nominal',
                                      'overall_protocols_io_doi': 'nominal',
                                      'parent_sample_id': 'nominal',
                                      'permeabilization_time_unit': 'nominal',
                                      'permeabilization_time_value': 'quantitative',
                                      'pi': 'nominal',
                                      'pi_email': 'nominal',
                                      'pixel_dwell_time_unit': 'nominal',
                                      'pixel_dwell_time_value': 'quantitative',
                                      'pixel_size_x_unit': 'nominal',
                                      'pixel_size_x_value': 'quantitative',
                                      'pixel_size_y_unit': 'nominal',
                                      'pixel_size_y_value': 'quantitative',
                                      'polarity': 'nominal',
                                      'preparation_instrument_kit': 'nominal',
                                      'preparation_instrument_model': 'nominal',
                                      'preparation_instrument_vendor': 'nominal',
                                      'preparation_maldi_matrix': 'nominal',
                                      'preparation_protocol_doi': 'nominal',
                                      'preparation_type': 'nominal',
                                      'primary_ion': 'nominal',
                                      'primary_ion_current_unit': 'nominal',
                                      'primary_ion_current_value': 'quantitative',
                                      'processing_protocols_io_doi': 'nominal',
                                      'processing_search': 'nominal',
                                      'protocols_io_doi': 'nominal',
                                      'published_timestamp': 'quantitative',
                                      'puck_id': 'nominal',
                                      'range_z_unit': 'nominal',
                                      'range_z_value': 'quantitative',
                                      'reagent_prep_protocols_io_doi': 'nominal',
                                      'resolution_x_unit': 'nominal',
                                      'resolution_x_value': 'quantitative',
                                      'resolution_y_unit': 'nominal',
                                      'resolution_y_value': 'quantitative',
                                      'resolution_z_unit': 'nominal',
                                      'resolution_z_value': 'quantitative',
                                      'rnaseq_assay_input': 'quantitative',
                                      'rnaseq_assay_input_unit': 'nominal',
                                      'rnaseq_assay_input_value': 'quantitative',
                                      'rnaseq_assay_method': 'nominal',
                                      'roi_description': 'nominal',
                                      'roi_id': 'quantitative',
                                      'sample_indexing_kit': 'nominal',
                                      'sample_indexing_set': 'nominal',
                                      'sample_quality_metric': 'nominal',
                                      'sc_isolation_cell_number': 'quantitative',
                                      'sc_isolation_enrichment': 'nominal',
                                      'sc_isolation_entity': 'nominal',
                                      'sc_isolation_protocols_io_doi': 'nominal',
                                      'sc_isolation_quality_metric': 'nominal',
                                      'sc_isolation_tissue_dissociation': 'nominal',
                                      'scan_direction': 'nominal',
                                      'section_prep_protocols_io_doi': 'nominal',
                                      'segment_data_format': 'nominal',
                                      'sequencing_batch_id': 'nominal',
                                      'sequencing_phix_percent': 'quantitative',
                                      'sequencing_read_format': 'nominal',
                                      'sequencing_read_percent_q30': 'quantitative',
                                      'sequencing_reagent_kit': 'nominal',
                                      'signal_type': 'nominal',
                                      'slide_id': 'nominal',
                                      'source_project': 'nominal',
                                      'source_storage_duration_unit': 'nominal',
                                      'source_storage_duration_value': 'quantitative',
                                      'spatial_sampling_type': 'nominal',
                                      'spatial_target': 'nominal',
                                      'spatial_type': 'nominal',
                                      'spot_size_unit': 'nominal',
                                      'spot_size_value': 'quantitative',
                                      'spot_spacing_unit': 'nominal',
                                      'spot_spacing_value': 'quantitative',
                                      'stain': 'nominal',
                                      'stain_name': 'nominal',
                                      'stain_technique': 'nominal',
                                      'start_datetime': 'nominal',
                                      'status': 'nominal',
                                      'step_z_value': 'quantitative',
                                      'tile_configuration': 'nominal',
                                      'tiled_image_columns': 'quantitative',
                                      'tiled_image_count': 'quantitative',
                                      'time_since_acquisition_instrument_calibration_unit': 'nominal',
                                      'time_since_acquisition_instrument_calibration_value': 'quantitative',
                                      'total_run_time_unit': 'nominal',
                                      'total_run_time_value': 'quantitative',
                                      'transposition_input': 'quantitative',
                                      'transposition_kit_number': 'nominal',
                                      'transposition_method': 'nominal',
                                      'transposition_reagent_kit': 'nominal',
                                      'transposition_transposase_source': 'nominal',
                                      'umi_offset': 'nominal',
                                      'umi_read': 'nominal',
                                      'umi_size': 'nominal',
                                      'uuid': 'nominal'},
                           'url': './data/hubmap_2025-05-05/datasets.csv'},
              'donors': {'fields': {'abo_blood_group_system': 'nominal',
                                    'age_unit': 'nominal',
                                    'age_value': 'quantitative',
                                    'apolipoprotein_e_phenotype': 'nominal',
                                    'body_mass_index_unit': 'nominal',
                                    'body_mass_index_value': 'quantitative',
                                    'cause_of_death': 'nominal',
                                    'created_by_user_displayname': 'nominal',
                                    'created_by_user_email': 'nominal',
                                    'created_timestamp': 'quantitative',
                                    'death_event': 'nominal',
                                    'egfr_ckd_epi_2021__unit': 'nominal',
                                    'egfr_ckd_epi_2021__value': 'quantitative',
                                    'ethnicity': 'nominal',
                                    'gestational_age_unit': 'nominal',
                                    'gestational_age_value': 'quantitative',
                                    'group_name': 'nominal',
                                    'height_unit': 'nominal',
                                    'height_value': 'quantitative',
                                    'hubmap_id': 'nominal',
                                    'kidney_donor_profile_index_unit': 'nominal',
                                    'kidney_donor_profile_index_value': 'quantitative',
                                    'last_modified_timestamp': 'quantitative',
                                    'mapped_consortium': 'nominal',
                                    'mechanism_of_injury': 'nominal',
                                    'medical_history': 'nominal',
                                    'other_anatomic_concept': 'nominal',
                                    'pathology_note': 'nominal',
                                    'race': 'nominal',
                                    'rh_blood_group': 'nominal',
                                    'serum_creatinine_unit': 'nominal',
                                    'serum_creatinine_value': 'quantitative',
                                    'sex': 'nominal',
                                    'social_history': 'nominal',
                                    'uuid': 'nominal',
                                    'weight_unit': 'nominal',
                                    'weight_value': 'quantitative'},
                         'url': './data/hubmap_2025-05-05/donors.csv'},
              'samples': {'fields': {'PPID': 'quantitative',
                                     'Preservation_condition': 'nominal',
                                     'Preservation_media': 'nominal',
                                     'area_unit': 'nominal',
                                     'area_value': 'quantitative',
                                     'cold_ischemia_time_unit': 'nominal',
                                     'cold_ischemia_time_value': 'quantitative',
                                     'created_by_user_displayname': 'nominal',
                                     'created_by_user_email': 'nominal',
                                     'created_timestamp': 'quantitative',
                                     'donor.hubmap_id': 'nominal',
                                     'entity_type': 'nominal',
                                     'group_name': 'nominal',
                                     'health_status': 'nominal',
                                     'hubmap_id': 'nominal',
                                     'is_suspension_enriched': 'nominal',
                                     'last_modified_timestamp': 'quantitative',
                                     'mapped_consortium': 'nominal',
                                     'metadata_schema_id': 'nominal',
                                     'notes': 'nominal',
                                     'organ_condition': 'nominal',
                                     'origin_samples_unique_mapped_organs': 'nominal',
                                     'pathologist_report': 'nominal',
                                     'perfusion_solution': 'nominal',
                                     'preparation_condition': 'nominal',
                                     'preparation_medium': 'nominal',
                                     'preparation_protocol_doi': 'nominal',
                                     'procedure_date': 'nominal',
                                     'processing_time_unit': 'nominal',
                                     'processing_time_value': 'quantitative',
                                     'quality_criteria': 'nominal',
                                     'sample_category': 'nominal',
                                     'sample_id': 'nominal',
                                     'section_index_number': 'quantitative',
                                     'source_id': 'nominal',
                                     'source_storage_duration_unit': 'nominal',
                                     'source_storage_duration_value': 'quantitative',
                                     'specimen_preservation_temperature': 'nominal',
                                     'specimen_quality_criteria': 'nominal',
                                     'specimen_tumor_distance_unit': 'nominal',
                                     'specimen_tumor_distance_value': 'quantitative',
                                     'storage_medium': 'nominal',
                                     'storage_method': 'nominal',
                                     'suspension_entity_count': 'quantitative',
                                     'suspension_entity_type': 'nominal',
                                     'thickness_unit': 'nominal',
                                     'thickness_value': 'quantitative',
                                     'tissue_weight_unit': 'nominal',
                                     'tissue_weight_value': 'quantitative',
                                     'uuid': 'nominal',
                                     'vital_state': 'nominal',
                                     'warm_ischemia_time_unit': 'nominal',
                                     'warm_ischemia_time_value': 'nominal'},
                          'url': './data/hubmap_2025-05-05/samples.csv'}},
 'relationships': [{'from_cardinality': 'many',
                    'from_entity': 'samples',
                    'from_field': 'donor.hubmap_id',
                    'to_cardinality': 'one',
                    'to_entity': 'donors',
                    'to_field': 'hubmap_id'},
                   {'from_cardinality': 'many',
                    'from_entity': 'datasets',
                    'from_field': 'donor.hubmap_id',
                    'to_cardinality': 'one',
                    'to_entity': 'donors',
                    'to_field': 'hubmap_id'},
                   {'from_cardinality': 'many',
                    'from_entity': 'datasets',
                    'from_field': 'donor.hubmap_id',
                    'to_cardinality': 'many',
                    'to_entity': 'samples',
                    'to_field': 'donor.hubmap_id'}]}


# Spec template strings (indexed by position)
TEMPLATES = ['{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F>"}, {"rollup": {"<E> count": '
 '{"op": "count"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "<F>", "type": '
 '"nominal"}, {"encoding": "y", "field": "<E> count", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F>"}, {"rollup": {"<E> count": '
 '{"op": "count"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "<E> count", "type": '
 '"quantitative"}, {"encoding": "y", "field": "<F>", "type": "nominal"}]}}',
 '{"source": [{"name": "<E1>", "source": "<E1.url>"}, {"name": "<E2>", "source": "<E2.url>"}], "transformation": '
 '[{"join": {"on": ["<E1.r.E2.id.from>", "<E1.r.E2.id.to>"]}, "in": ["<E1>", "<E2>"], "out": "<E1>__<E2>"}, '
 '{"groupby": "<E2.F>"}, {"rollup": {"<E1> count": {"op": "count"}}}], "representation": {"mark": "bar", "mapping": '
 '[{"encoding": "x", "field": "<E2.F>", "type": "nominal"}, {"encoding": "y", "field": "<E1> count", "type": '
 '"quantitative"}]}}',
 '{"source": [{"name": "<E1>", "source": "<E1.url>"}, {"name": "<E2>", "source": "<E2.url>"}], "transformation": '
 '[{"join": {"on": ["<E1.r.E2.id.from>", "<E1.r.E2.id.to>"]}, "in": ["<E1>", "<E2>"], "out": "<E1>__<E2>"}, '
 '{"groupby": "<E2.F>"}, {"rollup": {"<E1> count": {"op": "count"}}}], "representation": {"mark": "bar", "mapping": '
 '[{"encoding": "x", "field": "<E1> count", "type": "quantitative"}, {"encoding": "y", "field": "<E2.F>", "type": '
 '"nominal"}]}}',
 '{"source": [{"name": "<E1>", "source": "<E1.url>"}, {"name": "<E2>", "source": "<E2.url>"}], "transformation": '
 '[{"join": {"on": ["<E1.r.E2.id.from>", "<E1.r.E2.id.to>"]}, "in": ["<E1>", "<E2>"], "out": "<E1>__<E2>"}, '
 '{"groupby": ["<E2.F2>", "<E1.F1>"]}, {"rollup": {"count <E1>": {"op": "count"}}}], "representation": {"mark": "bar", '
 '"mapping": [{"encoding": "y", "field": "count <E1>", "type": "quantitative"}, {"encoding": "color", "field": '
 '"<E2.F2>", "type": "nominal"}, {"encoding": "x", "field": "<E1.F1>", "type": "nominal"}]}}',
 '{"source": [{"name": "<E1>", "source": "<E1.url>"}, {"name": "<E2>", "source": "<E2.url>"}], "transformation": '
 '[{"join": {"on": ["<E1.r.E2.id.from>", "<E1.r.E2.id.to>"]}, "in": ["<E1>", "<E2>"], "out": "<E1>__<E2>"}, '
 '{"groupby": ["<E2.F2>", "<E1.F1>"]}, {"rollup": {"count <E1>": {"op": "count"}}}], "representation": {"mark": "bar", '
 '"mapping": [{"encoding": "x", "field": "count <E1>", "type": "quantitative"}, {"encoding": "color", "field": '
 '"<E1.F1>", "type": "nominal"}, {"encoding": "y", "field": "<E2.F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": ["<F2>", "<F1>"]}, {"rollup": '
 '{"count <E>": {"op": "count"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "y", "field": "count '
 '<E>", "type": "quantitative"}, {"encoding": "color", "field": "<F1>", "type": "nominal"}, {"encoding": "x", "field": '
 '"<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": ["<F1>", "<F2>"]}, {"rollup": '
 '{"count <E>": {"op": "count"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "count '
 '<E>", "type": "quantitative"}, {"encoding": "color", "field": "<F1>", "type": "nominal"}, {"encoding": "y", "field": '
 '"<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": ["<F1>", "<F2>"]}, {"rollup": '
 '{"count <E>": {"op": "count"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "y", "field": "count '
 '<E>", "type": "quantitative"}, {"encoding": "xOffset", "field": "<F1>", "type": "nominal"}, {"encoding": "color", '
 '"field": "<F1>", "type": "nominal"}, {"encoding": "x", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": ["<F1>", "<F2>"]}, {"rollup": '
 '{"count <E>": {"op": "count"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "count '
 '<E>", "type": "quantitative"}, {"encoding": "yOffset", "field": "<F1>", "type": "nominal"}, {"encoding": "color", '
 '"field": "<F1>", "type": "nominal"}, {"encoding": "y", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>", "out": "groupCounts"}, '
 '{"rollup": {"<F2>_count": {"op": "count"}}}, {"groupby": ["<F1>", "<F2>"], "in": "<E>"}, {"rollup": '
 '{"<F1>_and_<F2>_count": {"op": "count"}}}, {"join": {"on": "<F2>"}, "in": ["<E>", "groupCounts"], "out": '
 '"datasets"}, {"derive": {"frequency": "d[\'<F1>_and_<F2>_count\'] / d[\'<F2>_count\']"}}], "representation": '
 '{"mark": "bar", "mapping": [{"encoding": "y", "field": "frequency", "type": "quantitative"}, {"encoding": "color", '
 '"field": "<F1>", "type": "nominal"}, {"encoding": "x", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>", "out": "groupCounts"}, '
 '{"rollup": {"<F2>_count": {"op": "count"}}}, {"groupby": ["<F1>", "<F2>"], "in": "<E>"}, {"rollup": '
 '{"<F1>_and_<F2>_count": {"op": "count"}}}, {"join": {"on": "<F2>"}, "in": ["<E>", "groupCounts"], "out": '
 '"datasets"}, {"derive": {"frequency": "d[\'<F1>_and_<F2>_count\'] / d[\'<F2>_count\']"}}], "representation": '
 '{"mark": "bar", "mapping": [{"encoding": "x", "field": "frequency", "type": "quantitative"}, {"encoding": "color", '
 '"field": "<F1>", "type": "nominal"}, {"encoding": "y", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"minimum <F1>": '
 '{"op": "min", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "minimum '
 '<F1>", "type": "quantitative"}, {"encoding": "y", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"minimum <F1>": '
 '{"op": "min", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "<F2>", '
 '"type": "nominal"}, {"encoding": "y", "field": "minimum <F1>", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"maximum <F1>": '
 '{"op": "max", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "maximum '
 '<F1>", "type": "quantitative"}, {"encoding": "y", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"maximum <F1>": '
 '{"op": "max", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "<F2>", '
 '"type": "nominal"}, {"encoding": "y", "field": "maximum <F1>", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"average <F1>": '
 '{"op": "mean", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": '
 '"average <F1>", "type": "quantitative"}, {"encoding": "y", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"average <F1>": '
 '{"op": "mean", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "<F2>", '
 '"type": "nominal"}, {"encoding": "y", "field": "average <F1>", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"median <F1>": '
 '{"op": "median", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": '
 '"median <F1>", "type": "quantitative"}, {"encoding": "y", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"median <F1>": '
 '{"op": "median", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": '
 '"<F2>", "type": "nominal"}, {"encoding": "y", "field": "median <F1>", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"total <F1>": '
 '{"op": "sum", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "total '
 '<F1>", "type": "quantitative"}, {"encoding": "y", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"total <F1>": '
 '{"op": "sum", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "<F2>", '
 '"type": "nominal"}, {"encoding": "y", "field": "total <F1>", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "representation": {"mark": "point", "mapping": [{"encoding": "x", '
 '"field": "<F1>", "type": "quantitative"}, {"encoding": "y", "field": "<F2>", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": ["<F1>", "<F2>"]}, {"rollup": '
 '{"count": {"op": "count"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "<F1>", '
 '"type": "nominal"}, {"encoding": "y", "field": "count", "type": "quantitative"}, {"encoding": "color", "field": '
 '"<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": ["<F1>", "<F2>"]}, {"rollup": '
 '{"count": {"op": "count"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "count", '
 '"type": "quantitative"}, {"encoding": "y", "field": "<F1>", "type": "nominal"}, {"encoding": "color", "field": '
 '"<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F>"}, {"rollup": {"frequency": '
 '{"op": "frequency"}}}], "representation": {"mark": "arc", "mapping": [{"encoding": "theta", "field": "frequency", '
 '"type": "quantitative"}, {"encoding": "color", "field": "<F>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F>"}, {"rollup": {"frequency": '
 '{"op": "frequency"}}}], "representation": {"mark": "arc", "mapping": [{"encoding": "theta", "field": "frequency", '
 '"type": "quantitative"}, {"encoding": "color", "field": "<F>", "type": "nominal"}, {"encoding": "radius", "value": '
 '60}, {"encoding": "radius2", "value": 80}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"rollup": {"<E> Records": {"op": "count"}}}]}',
 '{"source": {"name": "<E>", "source": "<E.url>"}}',
 '{"source": [{"name": "<E1>", "source": "<E1.url>"}, {"name": "<E2>", "source": "<E2.url>"}], "transformation": '
 '[{"join": {"on": ["<E1.r.E2.id.from>", "<E1.r.E2.id.to>"]}, "in": ["<E1>", "<E2>"], "out": "<E1>__<E2>"}]}',
 '{"source": [{"name": "<E1>", "source": "<E1.url>"}, {"name": "<E2>", "source": "<E2.url>"}], "transformation": '
 '[{"join": {"on": ["<E1.r.E2.id.from>", "<E1.r.E2.id.to>"]}, "in": ["<E1>", "<E2>"], "out": "<E1>__<E2>"}, '
 '{"groupby": "<E1.r.E2.id.from>"}, {"rollup": {"<E1> count": {"op": "count"}}}, {"orderby": {"field": "<E1> count", '
 '"order": "desc"}}, {"derive": {"rank": "rank()"}}, {"derive": {"most frequent": "d.rank == 1 ? \'yes\' : \'no\'"}}], '
 '"representation": [{"mark": "row", "mapping": [{"encoding": "x", "field": "<E1> count", "mark": "bar", "type": '
 '"quantitative", "domain": {"min": 0}}, {"encoding": "color", "column": "<E1> count", "mark": "bar", "field": "most '
 'frequent", "type": "nominal", "domain": ["yes", "no"], "range": ["#FFA500", "#c6cfd8"]}]}, {"mark": "row", '
 '"mapping": {"encoding": "text", "field": "*", "mark": "text", "type": "nominal"}}]}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\'] != null"}, {"orderby": '
 '{"field": "<F>", "order": "desc"}}, {"derive": {"largest": "rank() == 1 ? \'largest\' : \'not\'"}}], '
 '"representation": {"mark": "row", "mapping": [{"encoding": "x", "field": "<F>", "mark": "bar", "type": '
 '"quantitative"}, {"encoding": "color", "column": "<F>", "mark": "bar", "field": "largest", "type": "nominal", '
 '"domain": ["largest", "not"], "range": ["#FFA500", "c6cfd8"]}, {"encoding": "text", "field": "*", "mark": "text", '
 '"type": "nominal"}]}}',
 '{"source": [{"name": "<E1>", "source": "<E1.url>"}, {"name": "<E2>", "source": "<E2.url>"}], "transformation": '
 '[{"join": {"on": ["<E1.r.E2.id.from>", "<E1.r.E2.id.to>"]}, "in": ["<E1>", "<E2>"], "out": "<E1>__<E2>"}, '
 '{"groupby": "<E1.r.E2.id.from>"}, {"rollup": {"Largest <E1.F>": {"op": "max", "field": "<E1.F>"}}}, {"filter": '
 '"d[\'Largest <E1.F>\'] != null"}, {"orderby": {"field": "Largest <E1.F>", "order": "desc"}}, {"derive": {"rank": '
 '"rank()"}}, {"derive": {"largest": "d.rank == 1 ? \'yes\' : \'no\'"}}], "representation": {"mark": "row", "mapping": '
 '[{"encoding": "x", "field": "Largest <E1.F>", "mark": "bar", "type": "quantitative"}, {"encoding": "color", '
 '"column": "Largest <E1.F>", "mark": "bar", "field": "largest", "type": "nominal", "domain": ["yes", "no"], "range": '
 '["#FFA500", "#c6cfd8"]}, {"encoding": "text", "field": "*", "mark": "text", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\'] != null"}, {"orderby": '
 '{"field": "<F>", "order": "asc"}}, {"derive": {"smallest": "rank() == 1 ? \'smallest\' : \'not\'"}}], '
 '"representation": {"mark": "row", "mapping": [{"encoding": "color", "column": "<F>", "mark": "rect", "orderby": '
 '"<F>", "field": "smallest", "type": "nominal", "domain": ["smallest", "not"], "range": ["#ffdb9a", "white"]}, '
 '{"encoding": "text", "field": "*", "mark": "text", "type": "nominal"}]}}',
 '{"source": [{"name": "<E1>", "source": "<E1.url>"}, {"name": "<E2>", "source": "<E2.url>"}], "transformation": '
 '[{"join": {"on": ["<E1.r.E2.id.from>", "<E1.r.E2.id.to>"]}, "in": ["<E1>", "<E2>"], "out": "<E1>__<E2>"}, '
 '{"groupby": "<E1.r.E2.id.from>"}, {"rollup": {"Smallest <E1.F>": {"op": "min", "field": "<E1.F>"}}}, {"filter": '
 '"d[\'Smallest <E1.F>\'] != null"}, {"orderby": {"field": "Smallest <E1.F>", "order": "asc"}}, {"derive": {"rank": '
 '"rank()"}}, {"derive": {"smallest": "d.rank == 1 ? \'yes\' : \'no\'"}}], "representation": {"mark": "row", '
 '"mapping": [{"encoding": "color", "column": "Smallest <E1.F>", "mark": "bar", "orderby": "Smallest <E1.F>", "field": '
 '"smallest", "type": "nominal", "domain": ["yes", "no"], "range": ["#ffdb9a", "white"]}, {"encoding": "text", '
 '"field": "*", "mark": "text", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\'] != null"}, {"orderby": '
 '{"field": "<F>", "order": "asc"}}], "representation": {"mark": "row", "mapping": [{"encoding": "x", "column": "<F>", '
 '"mark": "bar", "field": "<F>", "type": "quantitative", "range": {"min": 0.2, "max": 1}}, {"encoding": "text", '
 '"field": "*", "mark": "text", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\'] != null"}, {"rollup": '
 '{"<F> min": {"op": "min", "field": "<F>"}, "<F> max": {"op": "max", "field": "<F>"}}}], "representation": {"mark": '
 '"row", "mapping": [{"encoding": "text", "field": "<F> min", "mark": "text", "type": "nominal"}, {"encoding": "text", '
 '"field": "<F> max", "mark": "text", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\'] != null"}, {"groupby": '
 '"<F>"}, {"rollup": {"count": {"op": "count"}}}], "representation": {"mark": "row", "mapping": [{"encoding": "text", '
 '"field": "<F>", "mark": "text", "type": "nominal"}, {"encoding": "x", "field": "count", "mark": "bar", "type": '
 '"quantitative", "range": {"min": 0.1, "max": 1}}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F1>\'] != null"}, {"groupby": '
 '"<F2>"}, {"rollup": {"<F1> min": {"op": "min", "field": "<F1>"}, "<F1> max": {"op": "max", "field": "<F1>"}}}, '
 '{"derive": {"range": "d[\'<F1> max\'] - d[\'<F1> min\']"}}, {"orderby": {"field": "range", "order": "desc"}}], '
 '"representation": {"mark": "row", "mapping": [{"encoding": "text", "field": "<F2>", "mark": "text", "type": '
 '"nominal"}, {"encoding": "text", "field": "<F1> min", "mark": "text", "type": "nominal"}, {"encoding": "x", '
 '"column": "range", "mark": "bar", "field": "<F1> min", "type": "quantitative", "domain": {"numberFields": ["<F1> '
 'min", "<F1> max"]}}, {"encoding": "x2", "column": "range", "mark": "bar", "field": "<F1> max", "type": '
 '"quantitative", "domain": {"numberFields": ["<F1> min", "<F1> max"]}}, {"encoding": "text", "field": "<F1> max", '
 '"mark": "text", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\']"}, {"groupby": "<F>"}, '
 '{"rollup": {"count": {"op": "count"}}}, {"orderby": {"field": "count", "order": "desc"}}, {"derive": {"rank": '
 '"rank()"}}, {"derive": {"most frequent": "d.rank == 1 ? \'yes\' : \'no\'"}}], "representation": {"mark": "row", '
 '"mapping": [{"encoding": "color", "column": "<F>", "mark": "bar", "orderby": "<F>", "field": "most frequent", '
 '"type": "nominal", "domain": ["yes", "no"], "range": ["#ffdb9a", "white"]}, {"encoding": "text", "field": "<F>", '
 '"mark": "text", "type": "nominal"}, {"encoding": "x", "field": "count", "mark": "bar", "type": "quantitative", '
 '"domain": {"min": 0}}, {"encoding": "color", "column": "count", "mark": "bar", "field": "most frequent", "type": '
 '"nominal", "domain": ["yes", "no"], "range": ["#FFA500", "#c6cfd8"]}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\'] != null"}, {"orderby": '
 '{"field": "<F>", "order": "asc"}}, {"derive": {"total": "count()"}}, {"derive": {"percentile": {"rolling": '
 '{"expression": "count() / d.total"}}}}], "representation": {"mark": "line", "mapping": [{"encoding": "x", "field": '
 '"<F>", "type": "quantitative"}, {"encoding": "y", "field": "percentile", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F1>\'] != null"}, {"orderby": '
 '{"field": "<F1>", "order": "asc"}}, {"groupby": "<F2>"}, {"derive": {"total": "count()"}}, {"derive": {"percentile": '
 '{"rolling": {"expression": "count() / d.total"}}}}], "representation": {"mark": "line", "mapping": [{"encoding": '
 '"x", "field": "<F1>", "type": "quantitative"}, {"encoding": "y", "field": "percentile", "type": "quantitative"}, '
 '{"encoding": "color", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": ["<F2>", "<F1>"]}, {"rollup": '
 '{"count <E>": {"op": "count"}}}, {"derive": {"udi_internal_percentile": "d[\'count <E>\'] / max(d[\'count '
 '<E>\'])"}}, {"derive": {"udi_internal_text_color_threshold": "d.udi_internal_percentile > .5 ? \'large\' : '
 '\'small\'"}}], "representation": [{"mark": "rect", "mapping": [{"encoding": "color", "field": "count <E>", "type": '
 '"quantitative"}, {"encoding": "y", "field": "<F1>", "type": "nominal"}, {"encoding": "x", "field": "<F2>", "type": '
 '"nominal"}]}, {"mark": "text", "mapping": [{"encoding": "text", "field": "count <E>", "type": "quantitative"}, '
 '{"encoding": "y", "field": "<F1>", "type": "nominal"}, {"encoding": "x", "field": "<F2>", "type": "nominal"}, '
 '{"encoding": "color", "field": "udi_internal_text_color_threshold", "type": "nominal", "domain": ["large", "small"], '
 '"range": ["white", "black"], "omitLegend": true}]}]}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": ["<F3>", "<F2>"]}, {"rollup": '
 '{"average <F1>": {"op": "mean", "field": "<F1>"}}}], "representation": {"mark": "rect", "mapping": [{"encoding": '
 '"color", "field": "average <F1>", "type": "quantitative"}, {"encoding": "y", "field": "<F2>", "type": "nominal"}, '
 '{"encoding": "x", "field": "<F3>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "representation": {"mark": "point", "mapping": [{"encoding": "x", '
 '"field": "<F1>", "type": "quantitative"}, {"encoding": "y", "field": "<F2>", "type": "quantitative"}, {"encoding": '
 '"color", "field": "<F3>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\'] != null"}, {"binby": '
 '{"field": "<F>", "output": {"bin_start": "start", "bin_end": "end"}}}, {"rollup": {"count": {"op": "count"}}}], '
 '"representation": {"mark": "rect", "mapping": [{"encoding": "x", "field": "start", "type": "quantitative"}, '
 '{"encoding": "x2", "field": "end", "type": "quantitative"}, {"encoding": "y", "field": "count", "type": '
 '"quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\'] != null"}, {"kde": '
 '{"field": "<F>", "output": {"sample": "<F>", "density": "density"}}}], "representation": {"mark": "area", "mapping": '
 '[{"encoding": "x", "field": "<F>", "type": "quantitative"}, {"encoding": "y", "field": "density", "type": '
 '"quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "representation": {"mark": "point", "mapping": {"encoding": "x", '
 '"field": "<F>", "type": "quantitative"}}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F1>\'] != null"}, {"groupby": '
 '"<F2>"}, {"kde": {"field": "<F1>", "output": {"sample": "<F1>", "density": "density"}}}], "representation": '
 '[{"mark": "area", "mapping": [{"encoding": "x", "field": "<F1>", "type": "quantitative"}, {"encoding": "color", '
 '"field": "<F2>", "type": "nominal"}, {"encoding": "y", "field": "density", "type": "quantitative"}, {"encoding": '
 '"opacity", "value": 0.25}]}, {"mark": "line", "mapping": [{"encoding": "x", "field": "<F1>", "type": '
 '"quantitative"}, {"encoding": "color", "field": "<F2>", "type": "nominal"}, {"encoding": "y", "field": "density", '
 '"type": "quantitative"}]}]}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "representation": {"mark": "point", "mapping": [{"encoding": "x", '
 '"field": "<F1>", "type": "quantitative"}, {"encoding": "y", "field": "<F2>", "type": "nominal"}, {"encoding": '
 '"color", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"derive": {"<E> Count": "count()"}}, {"filter": '
 '"d[\'<F>\'] != null"}, {"rollup": {"Valid <F> Count": {"op": "count"}, "<E> Count": {"op": "median", "field": "<E> '
 'Count"}}}, {"derive": {"Valid <F> %": "d[\'Valid <F> Count\'] / d[\'<E> Count\']"}}], "representation": {"mark": '
 '"row", "mapping": [{"encoding": "text", "field": "Valid <F> Count", "mark": "text", "type": "nominal"}, {"encoding": '
 '"text", "field": "<E> Count", "mark": "text", "type": "nominal"}, {"encoding": "x", "field": "Valid <F> %", "mark": '
 '"bar", "type": "quantitative", "domain": {"min": 0, "max": 1}}, {"encoding": "y", "field": "Valid <F> %", "mark": '
 '"line", "type": "quantitative", "range": {"min": 0.5, "max": 0.5}}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"derive": {"<E> Count": "count()"}}, {"filter": '
 '"d[\'<F>\'] != null"}, {"rollup": {"Valid <F> Count": {"op": "count"}, "<E> Count": {"op": "median", "field": "<E> '
 'Count"}}}, {"derive": {"Null <F> Count": "d[\'<E> Count\'] - d[\'Valid <F> Count\']", "Null <F> %": "1 - d[\'Valid '
 '<F> Count\'] / d[\'<E> Count\']"}}], "representation": {"mark": "row", "mapping": [{"encoding": "text", "field": '
 '"Null <F> Count", "mark": "text", "type": "nominal"}, {"encoding": "text", "field": "<E> Count", "mark": "text", '
 '"type": "nominal"}, {"encoding": "x", "field": "Null <F> %", "mark": "bar", "type": "quantitative", "domain": '
 '{"min": 0, "max": 1}}, {"encoding": "y", "field": "Null <F> %", "mark": "line", "type": "quantitative", "range": '
 '{"min": 0.5, "max": 0.5}}]}}']


# OpenAI function-calling tool definitions
TOOL_DEFS = [{'function': {'description': '[barchart] Counts entities grouped by a nominal field, displayed as a vertical bar '
                              'chart. Design: Vertical orientation chosen because category count is small (<=4), '
                              'keeping x-axis labels readable. Tasks: Compare counts across categories; identify the '
                              'most or least common category; assess the range of counts. Query patterns: How many <E> '
                              'are there, grouped by <F:n>?; Make a bar chart of <E> <F:n>.',
               'name': 'vis_000_barchart_count_vert_grouped',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[barchart] Counts entities grouped by a nominal field, displayed as a horizontal bar '
                              'chart. Design: Horizontal orientation chosen because category count is high (>4), '
                              'allowing longer labels on the y-axis. Tasks: Compare counts across categories; identify '
                              'the most or least common category; assess the range of counts. Query patterns: How many '
                              '<E> are there, grouped by <F:n>?; Make a bar chart of <E> <F:n>.',
               'name': 'vis_001_barchart_count_horiz_grouped',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[barchart] Joins two entities and counts records grouped by a field from the related '
                              'entity, displayed as a vertical bar chart. Design: Cross-entity join groups by a field '
                              'not native to the counted entity. Vertical orientation for small category counts (<=4). '
                              'Tasks: Compare counts across categories from a related entity; discover cross-entity '
                              'frequency patterns. Query patterns: How many <E1> are there, grouped by <E2.F:n>?',
               'name': 'vis_002_barchart_join_count_vert_grouped',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity1': {'description': 'The primary data entity (table).',
                                                         'type': 'string'},
                                             'entity2': {'description': 'The secondary data entity (table) to join '
                                                                        'with.',
                                                         'type': 'string'},
                                             'entity2_field': {'description': 'Field name (any type) from the entity.',
                                                               'type': 'string'}},
                              'required': ['entity1', 'entity2', 'entity2_field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[barchart] Joins two entities and counts records grouped by a field from the related '
                              'entity, displayed as a horizontal bar chart. Design: Cross-entity join with horizontal '
                              'orientation for higher category counts (>4). Tasks: Compare counts across categories '
                              'from a related entity; discover cross-entity frequency patterns. Query patterns: How '
                              'many <E1> are there, grouped by <E2.F:n>?',
               'name': 'vis_003_barchart_join_count_horiz_grouped',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity1': {'description': 'The primary data entity (table).',
                                                         'type': 'string'},
                                             'entity2': {'description': 'The secondary data entity (table) to join '
                                                                        'with.',
                                                         'type': 'string'},
                                             'entity2_field': {'description': 'Field name (any type) from the entity.',
                                                               'type': 'string'}},
                              'required': ['entity1', 'entity2', 'entity2_field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[stacked_bar] Joins two entities and produces a vertical stacked bar chart of counts '
                              'grouped by two nominal fields. Design: Stacked bars show part-to-whole composition '
                              'within each category. Vertical layout for small category counts (<=4). Color encodes '
                              'the secondary grouping field from the related entity. Tasks: Compare group compositions '
                              'across categories; identify dominant sub-groups within each bar. Query patterns: How '
                              'many <E1> are there, grouped by <E1.F1:n> and <E2.F2:n>?',
               'name': 'vis_004_stacked_bar_join_count_vert_stacked_grouped',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity1': {'description': 'The primary data entity (table).',
                                                         'type': 'string'},
                                             'entity1_field': {'description': 'Field name (any type) from the entity.',
                                                               'type': 'string'},
                                             'entity2': {'description': 'The secondary data entity (table) to join '
                                                                        'with.',
                                                         'type': 'string'},
                                             'entity2_field': {'description': 'Field name (any type) from the entity.',
                                                               'type': 'string'}},
                              'required': ['entity1', 'entity2', 'entity1_field', 'entity2_field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[stacked_bar] Joins two entities and produces a horizontal stacked bar chart of counts '
                              'grouped by two nominal fields. Design: Horizontal orientation for higher category '
                              'counts (>4). Color encodes the primary grouping field. Cross-entity join required. '
                              'Tasks: Compare group compositions across categories; identify dominant sub-groups '
                              'within each bar. Query patterns: How many <E1> are there, grouped by <E1.F1:n> and '
                              '<E2.F2:n>?',
               'name': 'vis_005_stacked_bar_join_count_horiz_stacked_grouped',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity1': {'description': 'The primary data entity (table).',
                                                         'type': 'string'},
                                             'entity1_field': {'description': 'Field name (any type) from the entity.',
                                                               'type': 'string'},
                                             'entity2': {'description': 'The secondary data entity (table) to join '
                                                                        'with.',
                                                         'type': 'string'},
                                             'entity2_field': {'description': 'Field name (any type) from the entity.',
                                                               'type': 'string'}},
                              'required': ['entity1', 'entity2', 'entity1_field', 'entity2_field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[stacked_bar] Counts entities grouped by two nominal fields, displayed as a vertical '
                              'stacked bar chart. Design: Vertical stacked layout for small category counts (<=4). '
                              'Color encodes the sub-group field; x-axis shows the primary grouping. Tasks: Compare '
                              'group compositions across categories; identify dominant sub-groups within each bar. '
                              'Query patterns: How many <E> are there, grouped by <F1:n> and <F2:n>?',
               'name': 'vis_006_stacked_bar_count_vert_stacked_grouped',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[stacked_bar] Counts entities grouped by two nominal fields, displayed as a horizontal '
                              'stacked bar chart. Design: Horizontal stacked layout for higher category counts (>4). '
                              'Color encodes the sub-group; stacking shows part-to-whole within each bar. Tasks: '
                              'Compare group compositions across categories; identify dominant sub-groups within each '
                              'bar. Query patterns: How many <E> are there, grouped by <F1:n> and <F2:n>?; What is the '
                              'count of <F1:n> for each <F2:n>?',
               'name': 'vis_007_stacked_bar_count_horiz_stacked_grouped',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[stacked_bar] Counts entities grouped by two nominal fields, displayed as a grouped '
                              '(side-by-side) vertical bar chart. Design: Uses xOffset for side-by-side grouping, '
                              'allowing direct comparison between sub-groups. Suitable for small category counts '
                              '(<=4). Tasks: Directly compare sub-group counts within and across categories. Query '
                              'patterns: What is the count of <F1:n> for each <F2:n>?',
               'name': 'vis_008_stacked_bar_count_vert_grouped',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[stacked_bar] Counts entities grouped by two nominal fields, displayed as a grouped '
                              '(side-by-side) horizontal bar chart. Design: Uses yOffset for side-by-side grouping in '
                              'horizontal orientation. Chosen when at least one field has more than 4 categories. '
                              'Tasks: Directly compare sub-group counts within and across categories. Query patterns: '
                              'What is the count of <F1:n> for each <F2:n>?',
               'name': 'vis_009_stacked_bar_count_horiz_grouped',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[stacked_bar] Shows the frequency (proportion) of one nominal field within each '
                              'category of another, as a vertical normalized bar chart. Design: Normalization computes '
                              'proportions per group, enabling fair comparison across groups of different sizes. '
                              'Vertical layout for small category counts (<=4). Tasks: Compare relative proportions '
                              'across categories; identify which sub-groups dominate in each group. Query patterns: '
                              'What is the frequency of <F1:n> for each <F2:n>?',
               'name': 'vis_010_stacked_bar_freq_vert_normalized',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[stacked_bar] Shows the frequency (proportion) of one nominal field within each '
                              'category of another, as a horizontal normalized bar chart. Design: Normalization for '
                              'proportional comparison. Horizontal layout for higher category counts (>4). Tasks: '
                              'Compare relative proportions across categories; identify which sub-groups dominate in '
                              'each group. Query patterns: What is the frequency of <F1:n> for each <F2:n>?',
               'name': 'vis_011_stacked_bar_freq_horiz_normalized',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[barchart] Computes the minimum of a quantitative field for each category, displayed as '
                              'a horizontal bar chart. Design: Horizontal orientation for many categories (>4). Bar '
                              'length encodes the minimum aggregate value for easy comparison. Tasks: Compare the '
                              'minimum value across categories; identify which group has the highest or lowest '
                              'minimum. Query patterns: What is the minimum <F1:q> for each <F2:n>?',
               'name': 'vis_012_barchart_min_horiz',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[barchart] Computes the minimum of a quantitative field for each category, displayed as '
                              'a vertical bar chart. Design: Vertical orientation for few categories (<=4). Bar height '
                              'encodes the minimum aggregate value. Tasks: Compare the minimum value across '
                              'categories; identify which group has the highest or lowest minimum. Query patterns: '
                              'What is the minimum <F1:q> for each <F2:n>?',
               'name': 'vis_013_barchart_min_vert',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[barchart] Computes the maximum of a quantitative field for each category, displayed as '
                              'a horizontal bar chart. Design: Horizontal orientation for many categories (>4). Bar '
                              'length encodes the maximum aggregate value for easy comparison. Tasks: Compare the '
                              'maximum value across categories; identify which group has the highest or lowest '
                              'maximum. Query patterns: What is the maximum <F1:q> for each <F2:n>?',
               'name': 'vis_014_barchart_max_horiz',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[barchart] Computes the maximum of a quantitative field for each category, displayed as '
                              'a vertical bar chart. Design: Vertical orientation for few categories (<=4). Bar height '
                              'encodes the maximum aggregate value. Tasks: Compare the maximum value across '
                              'categories; identify which group has the highest or lowest maximum. Query patterns: '
                              'What is the maximum <F1:q> for each <F2:n>?',
               'name': 'vis_015_barchart_max_vert',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[barchart] Computes the average of a quantitative field for each category, displayed as '
                              'a horizontal bar chart. Design: Horizontal orientation for many categories (>4). Bar '
                              'length encodes the average aggregate value for easy comparison. Tasks: Compare the '
                              'average value across categories; identify which group has the highest or lowest '
                              'average. Query patterns: What is the average <F1:q> for each <F2:n>?',
               'name': 'vis_016_barchart_avg_horiz',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[barchart] Computes the average of a quantitative field for each category, displayed as '
                              'a vertical bar chart. Design: Vertical orientation for few categories (<=4). Bar height '
                              'encodes the average aggregate value. Tasks: Compare the average value across '
                              'categories; identify which group has the highest or lowest average. Query patterns: '
                              'What is the average <F1:q> for each <F2:n>?',
               'name': 'vis_017_barchart_avg_vert',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[barchart] Computes the median of a quantitative field for each category, displayed as '
                              'a horizontal bar chart. Design: Horizontal orientation for many categories (>4). Bar '
                              'length encodes the median aggregate value for easy comparison. Tasks: Compare the '
                              'median value across categories; identify which group has the highest or lowest median. '
                              'Query patterns: What is the median <F1:q> for each <F2:n>?',
               'name': 'vis_018_barchart_median_horiz',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[barchart] Computes the median of a quantitative field for each category, displayed as '
                              'a vertical bar chart. Design: Vertical orientation for few categories (<=4). Bar height '
                              'encodes the median aggregate value. Tasks: Compare the median value across categories; '
                              'identify which group has the highest or lowest median. Query patterns: What is the '
                              'median <F1:q> for each <F2:n>?',
               'name': 'vis_019_barchart_median_vert',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[barchart] Computes the total of a quantitative field for each category, displayed as a '
                              'horizontal bar chart. Design: Horizontal orientation for many categories (>4). Bar '
                              'length encodes the total aggregate value for easy comparison. Tasks: Compare the total '
                              'value across categories; identify which group has the highest or lowest total. Query '
                              'patterns: What is the total <F1:q> for each <F2:n>?',
               'name': 'vis_020_barchart_sum_horiz',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[barchart] Computes the total of a quantitative field for each category, displayed as a '
                              'vertical bar chart. Design: Vertical orientation for few categories (<=4). Bar height '
                              'encodes the total aggregate value. Tasks: Compare the total value across categories; '
                              'identify which group has the highest or lowest total. Query patterns: What is the total '
                              '<F1:q> for each <F2:n>?',
               'name': 'vis_021_barchart_sum_vert',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[scatterplot] Plots two quantitative fields as a scatterplot to explore their '
                              'relationship. Design: Point marks on two quantitative axes reveal correlations, '
                              'clusters, and outliers. Data size capped at 100k rows for rendering performance. Tasks: '
                              'Assess correlation between two variables; identify clusters, outliers, extremes, and '
                              'the range of both variables. Query patterns: Is there a correlation between <F1:q> and '
                              '<F2:q>?; Make a scatterplot of <F1:q> and <F2:q>?',
               'name': 'vis_022_scatterplot_basic',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[stacked_bar] Creates a vertical stacked bar chart of counts grouped by two nominal '
                              'fields. Design: Vertical stacked layout for small primary category counts (<=4). Color '
                              'encodes the secondary field. Tasks: Compare group compositions across categories; '
                              'assess the overall range of counts. Query patterns: Make a stacked bar chart of <F1:n> '
                              'and <F2:n>?',
               'name': 'vis_023_stacked_bar_count_vert_stacked_grouped',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[stacked_bar] Creates a horizontal stacked bar chart of counts grouped by two nominal '
                              'fields. Design: Horizontal stacked layout for higher primary category counts (>4). '
                              'Color encodes the secondary field. Tasks: Compare group compositions across categories; '
                              'assess the overall range of counts. Query patterns: Make a stacked bar chart of <F1:n> '
                              'and <F2:n>?',
               'name': 'vis_024_stacked_bar_count_horiz_stacked_grouped',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[circular] Creates a pie chart showing the frequency distribution of a nominal field. '
                              'Design: Arc marks with theta encoding map frequency to angle. Suitable for fields with '
                              'few categories (<8) where part-to-whole perception is the goal. Tasks: Assess '
                              'part-to-whole proportions; identify the dominant category. Query patterns: Make a pie '
                              'chart of <F:n>?',
               'name': 'vis_025_circular_freq_distribution',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[circular] Creates a donut chart showing the frequency distribution of a nominal field. '
                              'Design: Donut variant with inner/outer radius creates a hollow center that can improve '
                              'label readability. Suitable for few categories (<8). Tasks: Assess part-to-whole '
                              'proportions; identify the dominant category. Query patterns: Make a donut chart of '
                              '<F:n>?',
               'name': 'vis_026_circular_freq_distribution',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Counts the total number of records in an entity and displays the result as a '
                              'single-row table. Design: Simple rollup with no visual encoding beyond the count value. '
                              'Useful as a quick data quality or size check. Tasks: Retrieve the total record count '
                              'for an entity. Query patterns: How many <E> records are there?',
               'name': 'vis_027_table_count',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Displays the raw data for an entity as a table. Design: No aggregation or '
                              'transformation applied; shows the underlying data as-is for exploration. Tasks: Explore '
                              'raw data; retrieve specific values; understand field values and ranges; identify '
                              'anomalies and extremes. Query patterns: What does the <E> data look like?; Make a table '
                              'of <E>?',
               'name': 'vis_028_table_raw',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Joins two related entities and displays the combined data as a table. Design: '
                              'Cross-entity join enriches the view by combining fields from two related entities. '
                              'Requires a valid foreign-key relationship. Tasks: Explore combined data from two '
                              'related entities; retrieve specific values; identify anomalies and extremes. Query '
                              'patterns: What does the combined data of <E1> and <E2> look like?; Make a table that '
                              'combines <E1> and <E2>.',
               'name': 'vis_029_table_join',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity1': {'description': 'The primary data entity (table).',
                                                         'type': 'string'},
                                             'entity2': {'description': 'The secondary data entity (table) to join '
                                                                        'with.',
                                                         'type': 'string'}},
                              'required': ['entity1', 'entity2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Finds which related entity record has the highest count of associated records, '
                              'displayed as a ranked table with bar indicators. Design: Groups by foreign key, counts, '
                              'ranks, and highlights the top record with color encoding. Bar marks on the count column '
                              'provide visual comparison. Tasks: Identify the record with the most associated '
                              'entities; compare counts across records. Query patterns: What <E2> has the most <E1>?',
               'name': 'vis_030_table_join_count_ranked',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity1': {'description': 'The primary data entity (table).',
                                                         'type': 'string'},
                                             'entity2': {'description': 'The secondary data entity (table) to join '
                                                                        'with.',
                                                         'type': 'string'}},
                              'required': ['entity1', 'entity2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Finds the record with the largest value in a quantitative field, displayed as a '
                              'ranked table with bar indicators. Design: Sorts descending by the target field, derives '
                              'a rank, and highlights the top record with color. Bar marks provide visual magnitude '
                              'comparison. Tasks: Identify the record with the largest value; compare values across '
                              'records. Query patterns: What Record in <E> has the largest <F:q>?',
               'name': 'vis_031_table_ranked',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Joins two entities, computes the maximum of a quantitative field per group, and '
                              'ranks the results in a table with bar indicators. Design: Cross-entity join followed by '
                              'group-level max aggregation. Highlights the top record with color encoding. Tasks: '
                              'Identify which related record has the largest aggregated value; compare across groups. '
                              'Query patterns: What Record in <E2> has the largest <E1> <E1.F:q>?',
               'name': 'vis_032_table_join_max_ranked',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity1': {'description': 'The primary data entity (table).',
                                                         'type': 'string'},
                                             'entity1_field': {'description': 'Field name (any type) from the entity.',
                                                               'type': 'string'},
                                             'entity2': {'description': 'The secondary data entity (table) to join '
                                                                        'with.',
                                                         'type': 'string'}},
                              'required': ['entity1', 'entity2', 'entity1_field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Finds the record with the smallest value in a quantitative field, displayed as '
                              'a ranked table with conditional formatting. Design: Sorts ascending by the target '
                              'field, derives a rank, and highlights the top record with background color. Uses rect '
                              'mark for row-level highlighting. Tasks: Identify the record with the smallest value; '
                              'compare values across records. Query patterns: What Record in <E> has the smallest '
                              '<F:q>?',
               'name': 'vis_033_table_ranked',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Joins two entities, computes the minimum of a quantitative field per group, and '
                              'ranks the results in a table with conditional formatting. Design: Cross-entity join '
                              'followed by group-level min aggregation. Highlights the top record with background '
                              'color via rect mark. Tasks: Identify which related record has the smallest aggregated '
                              'value; compare across groups. Query patterns: What Record in <E2> has the smallest <E1> '
                              '<E1.F:q>?',
               'name': 'vis_034_table_join_min_ranked',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity1': {'description': 'The primary data entity (table).',
                                                         'type': 'string'},
                                             'entity1_field': {'description': 'Field name (any type) from the entity.',
                                                               'type': 'string'},
                                             'entity2': {'description': 'The secondary data entity (table) to join '
                                                                        'with.',
                                                         'type': 'string'}},
                              'required': ['entity1', 'entity2', 'entity1_field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Sorts entity records by a quantitative field and displays the result as an '
                              'ordered table with in-cell bar marks. Design: Ordered by the quantitative field with '
                              'nulls filtered out. In-cell bar marks provide visual comparison of magnitude alongside '
                              'the text values. Tasks: View records in sorted order; compare relative magnitudes. '
                              'Query patterns: Order the <E> by <F:q>?',
               'name': 'vis_035_table_sorted',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Computes the minimum and maximum of a quantitative field and displays them as a '
                              'single-row table. Design: Simple rollup of min and max. Filters out nulls before '
                              'aggregation for accuracy. Tasks: Determine the range of a quantitative field. Query '
                              'patterns: What is the range of <E> <F:q> values?',
               'name': 'vis_036_table_min',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Lists all distinct values of a nominal field with their counts, displayed as a '
                              'table with in-cell bar marks. Design: Groups by the nominal field and counts '
                              'occurrences. In-cell bars provide visual frequency comparison. Tasks: Determine the '
                              'range (distinct values) of a nominal field; compare category frequencies. Query '
                              'patterns: What is the range of <E> <F:n> values?',
               'name': 'vis_037_table_count_distinct',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Computes the min and max of a quantitative field for each category of a nominal '
                              'field, displayed as a table with range bar marks. Design: Groups by nominal field, '
                              'computes min/max and derived range, then orders by range descending. Uses x/x2 encoding '
                              'to show the span between min and max values. Tasks: Compare the spread of a '
                              'quantitative field across categories; identify which group has the widest or narrowest '
                              'range. Query patterns: What is the range of <E> <F1:q> values for every <F2:n>?',
               'name': 'vis_038_table_range',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Finds the most frequent value of a nominal field, displayed as a ranked table '
                              'with bar marks and conditional formatting. Design: Groups by nominal field, counts, '
                              'ranks, and highlights the top value. Combines bar marks for count comparison and '
                              'background color for emphasis. Tasks: Identify the most frequent category; compare '
                              'frequencies across all categories. Query patterns: What is the most frequent <F:n>?',
               'name': 'vis_039_table_ranked_mode',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[line] Shows the cumulative distribution function (CDF) of a quantitative field as a '
                              'line chart. Design: Sorts by value, computes rolling percentile, and draws a line. The '
                              'CDF reveals the full distribution shape including median, quartiles, and tails. Tasks: '
                              'Characterize the distribution of a variable; identify median, quartiles, and '
                              'concentration of values. Query patterns: What is the cumulative distribution of <F:q>?; '
                              'Make a CDF plot of <F:q>.',
               'name': 'vis_040_line_cdf',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[grouped_line] Shows the cumulative distribution of a quantitative field for each '
                              'category of a nominal field, with separate lines per group. Design: Groups by nominal '
                              'field before computing per-group CDF. Color encodes group identity. Limited to fewer '
                              'than 5 groups for readability. Tasks: Compare distributions across groups; identify '
                              'which groups have higher or lower concentrations of values. Query patterns: What is the '
                              'cumulative distribution of <F1:q> for each <F2:n>?; Make a CDF plot of <F1:q> with a '
                              'line for each <F2:n>.',
               'name': 'vis_041_grouped_line_cdf',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[heatmap] Displays the count of entities for each combination of two nominal fields as '
                              'a heatmap with labeled cells. Design: Rect marks with quantitative color encoding show '
                              'density. Overlaid text marks display exact counts. Text color adapts based on cell '
                              'intensity for readability. Tasks: Identify clusters or patterns in the co-occurrence of '
                              'two fields; compare counts across combinations; find correlations. Query patterns: Are '
                              'there any clusters with respect to <E> counts of <F1:n> and <F2:n>?; Make a heatmap of '
                              '<E> <F1:n> and <F2:n>.',
               'name': 'vis_042_heatmap_count',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[heatmap] Displays the average of a quantitative field for each combination of two '
                              'nominal fields as a heatmap. Design: Uses three fields: a quantitative measure '
                              'aggregated by average, and two nominal axes. Color encodes the aggregate value. Tasks: '
                              'Identify patterns in the average value across two categorical dimensions; find '
                              'combinations with extreme values. Query patterns: What is the average <F1:q> for each '
                              '<F2:n> and <F3:n>?',
               'name': 'vis_043_heatmap_avg',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field3': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2', 'field3'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[grouped_scatter] Plots two quantitative fields as a scatterplot with points colored by '
                              'a nominal field to reveal group-level clusters. Design: Adds color encoding to a '
                              'standard scatterplot to separate groups visually. Limited to fewer than 8 color '
                              'categories for perceptual clarity. Tasks: Identify clusters that separate by group; '
                              'assess whether the relationship between two quantitative fields differs across groups. '
                              'Query patterns: Are there clusters of <E> <F1:q> and <F2:q> values across different '
                              '<F3:n> groups?',
               'name': 'vis_044_grouped_scatter_by_color',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field3': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2', 'field3'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[histogram] Shows the distribution of a quantitative field as a histogram with '
                              'automatically computed bins. Design: Uses binby to create equal-width bins. Rect marks '
                              'span from bin start to bin end on x, with count on y. Tasks: Characterize the shape of '
                              'a distribution; identify modes, skewness, and gaps. Query patterns: What is the '
                              'distribution of <F:q>?; Make a histogram of <F:q>?',
               'name': 'vis_045_histogram_distribution',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[area] Shows the distribution of a quantitative field as a smooth density curve (KDE) '
                              'rendered as an area chart. Design: Kernel density estimation produces a smooth curve. '
                              'Area mark fills below the density line. Used for moderate cardinality (50-250) where a '
                              'smooth estimate is more informative than binning. Tasks: Characterize the shape of a '
                              'distribution; identify modes and overall density patterns. Query patterns: What is the '
                              'distribution of <F:q>?',
               'name': 'vis_046_area_density',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[dot] Shows the distribution of a quantitative field as individual points along a '
                              'single axis. Design: Point marks on a single quantitative x-axis. Best for small '
                              'datasets (50 or fewer values) where individual observations are meaningful and '
                              'overplotting is minimal. Tasks: Characterize the distribution; identify individual '
                              'values, clusters, and outliers. Query patterns: What is the distribution of <F:q>?',
               'name': 'vis_047_dot_distribution',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[grouped_area] Compares the distribution of a quantitative field across categories '
                              'using overlapping density curves (KDE) with area and line marks. Design: Per-group KDE '
                              'with semi-transparent area fills and line outlines. Color encodes group identity. '
                              'Limited to fewer than 4 groups to avoid excessive overlap. Opacity set to 0.25 for '
                              'layering. Tasks: Compare distribution shapes across groups; identify shifts in central '
                              'tendency or spread. Query patterns: Is the distribution of <F1:q> similar for each '
                              '<F2:n>?',
               'name': 'vis_048_grouped_area_density',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[grouped_dot] Compares the distribution of a quantitative field across categories using '
                              'dot strips, with one row per category. Design: Points plotted on a quantitative x-axis '
                              'with nominal y-axis for group separation. Color reinforces group identity. Best for '
                              'small datasets (50 or fewer values per group). Tasks: Compare distributions across '
                              'groups; identify clusters and outliers within each group. Query patterns: Is the '
                              'distribution of <F1:q> similar for each <F2:n>?',
               'name': 'vis_049_grouped_dot_distribution',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Analyzes data completeness by counting and computing the percentage of records '
                              'with non-null values in a specified field. Design: Derives total count before '
                              'filtering, then computes valid count and percentage. Percentage bar with 50% reference '
                              'line provides visual context for data completeness. Tasks: Assess data completeness for '
                              'a field; determine how many records have valid values and what proportion. Query '
                              'patterns: How many <E> records have a non-null <F:q|o|n>?; What percentage of <E> '
                              'records have a non-null <F:q|o|n>?',
               'name': 'vis_050_table_count_null_nonnull',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Analyzes data quality by counting and computing the percentage of records with '
                              'null values in a specified field. Design: Derives null count as total minus valid '
                              'count. Percentage bar shows the null proportion with a 50% reference line. Tasks: '
                              'Assess data quality; determine how many records are missing a value and what '
                              'proportion. Query patterns: How many <E> records have a null <F:q|o|n>?; What '
                              'percentage of <E> records have a null <F:q|o|n>?',
               'name': 'vis_051_table_count_null',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'}]


# Dispatch: tool name -> (template_index, param_to_binding_map)
TOOL_DISPATCH = {'vis_000_barchart_count_vert_grouped': (0, {'entity': 'E', 'field': 'F'}),
 'vis_001_barchart_count_horiz_grouped': (1, {'entity': 'E', 'field': 'F'}),
 'vis_002_barchart_join_count_vert_grouped': (2, {'entity1': 'E1', 'entity2': 'E2', 'entity2_field': 'E2.F'}),
 'vis_003_barchart_join_count_horiz_grouped': (3, {'entity1': 'E1', 'entity2': 'E2', 'entity2_field': 'E2.F'}),
 'vis_004_stacked_bar_join_count_vert_stacked_grouped': (4,
                                                         {'entity1': 'E1',
                                                          'entity1_field': 'E1.F1',
                                                          'entity2': 'E2',
                                                          'entity2_field': 'E2.F2'}),
 'vis_005_stacked_bar_join_count_horiz_stacked_grouped': (5,
                                                          {'entity1': 'E1',
                                                           'entity1_field': 'E1.F1',
                                                           'entity2': 'E2',
                                                           'entity2_field': 'E2.F2'}),
 'vis_006_stacked_bar_count_vert_stacked_grouped': (6, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_007_stacked_bar_count_horiz_stacked_grouped': (7, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_008_stacked_bar_count_vert_grouped': (8, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_009_stacked_bar_count_horiz_grouped': (9, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_010_stacked_bar_freq_vert_normalized': (10, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_011_stacked_bar_freq_horiz_normalized': (11, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_012_barchart_min_horiz': (12, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_013_barchart_min_vert': (13, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_014_barchart_max_horiz': (14, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_015_barchart_max_vert': (15, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_016_barchart_avg_horiz': (16, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_017_barchart_avg_vert': (17, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_018_barchart_median_horiz': (18, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_019_barchart_median_vert': (19, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_020_barchart_sum_horiz': (20, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_021_barchart_sum_vert': (21, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_022_scatterplot_basic': (22, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_023_stacked_bar_count_vert_stacked_grouped': (23, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_024_stacked_bar_count_horiz_stacked_grouped': (24, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_025_circular_freq_distribution': (25, {'entity': 'E', 'field': 'F'}),
 'vis_026_circular_freq_distribution': (26, {'entity': 'E', 'field': 'F'}),
 'vis_027_table_count': (27, {'entity': 'E'}),
 'vis_028_table_raw': (28, {'entity': 'E'}),
 'vis_029_table_join': (29, {'entity1': 'E1', 'entity2': 'E2'}),
 'vis_030_table_join_count_ranked': (30, {'entity1': 'E1', 'entity2': 'E2'}),
 'vis_031_table_ranked': (31, {'entity': 'E', 'field': 'F'}),
 'vis_032_table_join_max_ranked': (32, {'entity1': 'E1', 'entity1_field': 'E1.F', 'entity2': 'E2'}),
 'vis_033_table_ranked': (33, {'entity': 'E', 'field': 'F'}),
 'vis_034_table_join_min_ranked': (34, {'entity1': 'E1', 'entity1_field': 'E1.F', 'entity2': 'E2'}),
 'vis_035_table_sorted': (35, {'entity': 'E', 'field': 'F'}),
 'vis_036_table_min': (36, {'entity': 'E', 'field': 'F'}),
 'vis_037_table_count_distinct': (37, {'entity': 'E', 'field': 'F'}),
 'vis_038_table_range': (38, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_039_table_ranked_mode': (39, {'entity': 'E', 'field': 'F'}),
 'vis_040_line_cdf': (40, {'entity': 'E', 'field': 'F'}),
 'vis_041_grouped_line_cdf': (41, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_042_heatmap_count': (42, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_043_heatmap_avg': (43, {'entity': 'E', 'field1': 'F1', 'field2': 'F2', 'field3': 'F3'}),
 'vis_044_grouped_scatter_by_color': (44, {'entity': 'E', 'field1': 'F1', 'field2': 'F2', 'field3': 'F3'}),
 'vis_045_histogram_distribution': (45, {'entity': 'E', 'field': 'F'}),
 'vis_046_area_density': (46, {'entity': 'E', 'field': 'F'}),
 'vis_047_dot_distribution': (47, {'entity': 'E', 'field': 'F'}),
 'vis_048_grouped_area_density': (48, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_049_grouped_dot_distribution': (49, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_050_table_count_null_nonnull': (50, {'entity': 'E', 'field': 'F'}),
 'vis_051_table_count_null': (51, {'entity': 'E', 'field': 'F'})}
