export interface GcpConfig {
    project_id: string
    location: string
}

export interface SourcesConfig {
    calpads?: string
    nwea?: string
    iready?: string
    star?: string
    cers?: string
    iab?: string
    [key: string]: string | undefined
}

export interface SourcesSuffixConfig {
    cers?: string
    nwea?: string
    iready?: string
    star?: string
    [key: string]: string | undefined
}

export interface SourcesOverridesConfig {
    [key: string]: string
}

export interface ExcludeColsConfig {
    nwea?: string[]
    star?: string[]
    iready?: string[]
    [key: string]: string[] | undefined
}

export interface OptionsConfig {
    cache_csv?: boolean
}

export interface PathsConfig {
    data_dir?: string
    charts_dir?: string
    output_dir?: string
    config_dir?: string
}

export interface PartnerConfig {
    partner_name: string
    district_name?: string[]
    gcp: GcpConfig
    sources: SourcesConfig
    sources_suffix?: SourcesSuffixConfig
    sources_overrides?: SourcesOverridesConfig
    exclude_cols?: ExcludeColsConfig
    options?: OptionsConfig
    paths?: PathsConfig
    school_name_map?: Record<string, string>
    [key: string]: unknown
}

export interface BaseSettings {
    partner_name: string
    [key: string]: unknown
}
