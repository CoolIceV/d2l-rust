use tch::{nn, nn::OptimizerConfig, Tensor};
use polars::{prelude::*, export::rayon::str::EncodeUtf16};

fn main() {
   let df = CsvReader::from_path("iris.csv").unwrap()
                        .has_header(true)
                        .finish().unwrap();
    df.fill_null(FillNullStrategy::Mean);
    df.column("name");
    
    let columns_to_encode = vec![
        "MSZoning",
        "Street",
        // 添加其他需要进行One-Hot编码的列名
    ];
    
    let mut df_encoded = df.clone();
    
    for col_name in columns_to_encode {
        let col = df.column(col_name).unwrap();
        
        let unique_values = col.unique().unwrap();
        
        for i in 0..unique_values.len() {
            let value = unique_values.get(i).unwrap().to_string();
            let new_col_name = format!("{}_{}", col_name, value);
            
            let new_col = (df.with_column(col_name) == 1).alias(&new_col_name);
            
            df_encoded = df_encoded
                .with_column(new_col)
                .unwrap();
        }
        
        df_encoded = df_encoded
            .drop(col_name)
            .unwrap();
    }
    
    println!("{}", df_encoded);
}
