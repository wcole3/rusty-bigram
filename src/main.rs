use std::collections::HashMap;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;

/**
    * A simple rust script implementing a bigram language model
 **/

fn main() {
    println!(" Welcome to the bigram name model!");

    //load in the names file
    let names = include_str!("../files/names.txt");
    // split on new lines
    let names: Vec<&str> = names.split("\n").collect();
    let cleaned_names: Vec<String> = names.iter().map(|name| clean_name(name)).collect();
    // now we need to restructure this into a matrix of bigram counts
    // we can then use this to calculate the probability of each bigram
    // and then use this to generate new names
    let bigram_matrix = create_bigram_matrix(&cleaned_names, 1.0);
    //println!("{:?}", &bigram_matrix[..3]);
    // print the first few names and the neg log likelihood
    for name in &cleaned_names[..5] {
        println!("name: {}, -log(likelihood): {}", name, -1.0*likelihood_of_word(name, &bigram_matrix).log10()/(name.len() as f64));
    }

    // Sample the matrix a few times
    for _ in 0..5 {
        let mut name = String::new();
        let mut current_char = 0;
        name.push(int_to_char(current_char));
        loop {
            let next_char = sample_next_char(&bigram_matrix[current_char]);
            current_char = next_char;
            name.push(int_to_char(next_char));
            if int_to_char(next_char) == '.' {
                break;
            }
        }
        println!("Generated name: {}, -log(likelihood): {}", name, -1.0*likelihood_of_word(&name, &bigram_matrix).log10()/(name.len() as f64));
    }
}

/**
    * Function to clean the names
    * 1. Remove any non-alphabetic characters
    * 2. Convert to lowercase
    * 3. Add a dot to the start and end of the name
**/
fn clean_name(name: &str) -> String {
    // remove any non-alphabetic characters
    let name: String = name.chars().filter(|c| c.is_alphabetic()).collect();
    // convert to lowercase
    let name = name.to_lowercase();
    // add dot to start and end
    let name = format!(".{}.", name);
    name
}

/**
    * Function to count the bigrams
    * 1. Create a hashmap to store the bigram counts
    * 2. Iterate over the names
    * 3. For each name, iterate over the characters
    * 4. For each character, get the bigram and increment the count
**/
fn count_bigrams(names: &Vec<String>) -> HashMap<String, i32> {
    let mut bigram_counts = HashMap::new();
    for name in &names[..] {
        for i in 0..name.len() - 1 {
            let bigram = &name[i..i+2];
            let count = bigram_counts.entry(bigram.to_string()).or_insert(0);
            *count += 1;
        }
    }
    bigram_counts
}

/**
    * Function to create the bigram matrix
    * 1. Create a matrix of zeros
    * 2. For each character starting and ending with the dot character, create a row for each bigram starting with that character
    * 3. For each bigram, calculate the probability of that bigram by dividing the count by the total number of bigrams starting with that character
**/
fn create_bigram_matrix(names: &Vec<String>, smoothing: f64) -> Vec<Vec<f64>> {
    let mut bigram_matrix = vec![vec![smoothing; 27]; 27];
    let mut bigram_totals = vec![smoothing as i64; 27];
    // have to reset first index to 0.0; we never want a dot-dot bigram to be generated
    bigram_totals[0] = 0;
    bigram_matrix[0][0] = 0.0;
    for name in &names[..] {
        for i in 0..name.len() - 1 {
            let bigram = &name[i..i+2];
            // need to check for dot character
            let mut first = 0;
            let mut second = 0;
            let chars = bigram.chars().collect::<Vec<char>>();
            if chars[0] != '.' {
                first = chars[0] as usize - 96;
            }
            if chars[1] != '.' {
                second = chars[1] as usize - 96;
            }
            bigram_matrix[first][second] += 1.0;
            bigram_totals[first] += 1;
        }
    }
    for i in 0..27 {
        for j in 0..27 {
            bigram_matrix[i][j] /= bigram_totals[i] as f64;
        }
    }
    bigram_matrix
}

/**
    * Function to take in a row of the porbablity matrix
    * and sample it as a multinomial distribution to return
    * the nindex of the next character
**/
fn sample_next_char(probablities: &Vec<f64>) -> usize {
    // weighted index dist
    let dist = WeightedIndex::new(probablities).unwrap();
    // TODO might want to used seeded rng
    // https://rust-random.github.io/rand/rand_core/trait.SeedableRng.html
    let mut rng = thread_rng();
    dist.sample(&mut rng)
}

/**
    * Function to convert an int to a char
**/
fn int_to_char(index: usize) -> char {
    if index == 0 {
        return '.';
    }
    (index as u8 + 96) as char
}

/**
    * Cacluate the likelihood of a word from the bigram matrix
**/
fn likelihood_of_word(word: &str, bigram_matrix: &Vec<Vec<f64>>) -> f64 {
    let mut likelihood = 1.0;
    for i in 0..word.len() - 1 {
        let bigram = &word[i..i+2];
        let mut first = 0;
        let mut second = 0;
        let chars = bigram.chars().collect::<Vec<char>>();
        if chars[0] != '.' {
            first = chars[0] as usize - 96;
        }
        if chars[1] != '.' {
            second = chars[1] as usize - 96;
        }
        likelihood *= bigram_matrix[first][second];
    }
    likelihood
}


