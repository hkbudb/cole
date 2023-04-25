use std::cmp::{Ordering, PartialOrd, PartialEq};
use super::types::{CompoundKey, COMPOUND_KEY_SIZE};
use rug::{Rational, Integer, Complete, integer::Order};
use anyhow::{Result, anyhow};

/* Re-implement the piecewise geometrical model from: https://github.com/gvinciguerra/PGM-index/blob/master/include/pgm/piecewise_linear_model.hpp
 */
#[derive(Debug, Clone)]
pub struct Slope {
    dx: Integer,
    dy: i64,
}

impl PartialOrd for Slope {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if (&self.dy * &other.dx).complete() == (&self.dx * &other.dy).complete() {
            Some(Ordering::Equal)
        }
        else if (&self.dy * &other.dx).complete() < (&self.dx * &other.dy).complete() {
            Some(Ordering::Less)
        }
        else {
            Some(Ordering::Greater)
        }
    }

    fn lt(&self, other: &Self) -> bool {
        (&self.dy * &other.dx).complete() < (&self.dx * &other.dy).complete()
    }

    fn le(&self, other: &Self) -> bool {
        (&self.dy * &other.dx).complete() <= (&self.dx * &other.dy).complete()
    }

    fn gt(&self, other: &Self) -> bool {
        (&self.dy * &other.dx).complete() > (&self.dx * &other.dy).complete()
    }

    fn ge(&self, other: &Self) -> bool {
        (&self.dy * &other.dx).complete() >= (&self.dx * &other.dy).complete()
    }
}

impl PartialEq for Slope {
    fn eq(&self, other: &Self) -> bool {
        (&self.dy * &other.dx).complete() == (&self.dx * &other.dy).complete()
    }

    fn ne(&self, other: &Self) -> bool {
        (&self.dy * &other.dx).complete() != (&self.dx * &other.dy).complete()
    }
}

/* Each point is made up of a big integer and the corresopnding position in the data stream
 */
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    pub x: Integer,
    pub y: i64,
}

/* Define the substract of two points
 */
impl<'a> std::ops::Sub for &'a Point {
    type Output = Slope;
    fn sub(self, rhs: Self) -> Self::Output {
        Slope {
            dx: (&self.x - &rhs.x).complete(),
            dy: self.y - rhs.y,
        }
    }
}

/* Raw PGM Generator
 */
#[derive(Debug, Clone, PartialEq)]
pub struct OptimalPiecewiseLinearModel {
    epsilon: i64,
    lower: Vec<Point>,
    upper: Vec<Point>,
    first_x: Integer,
    first_y: i64,
    last_x: Integer,
    last_y: i64,
    lower_start: usize,
    upper_start: usize,
    points_in_hull: usize,
    rectangle: [Point; 4],
}

#[allow(non_snake_case)]
fn cross(O: &Point, A: &Point, B: &Point) -> Integer {
    let OA = A - O;
    let OB = B - O;
    OA.dx * OB.dy - OA.dy * OB.dx
}

impl OptimalPiecewiseLinearModel {
    pub fn new(epsilon: i64) -> Self {
        Self {
            epsilon: epsilon,
            lower: Vec::<Point>::new(),
            upper: Vec::<Point>::new(),
            first_x: Integer::new(),
            first_y: 0,
            last_x: Integer::new(),
            last_y: 0,
            lower_start: 0,
            upper_start: 0,
            points_in_hull: 0,
            rectangle: [Point{x: Integer::new(), y:0}, Point{x: Integer::new(), y:0}, Point{x: Integer::new(), y:0}, Point{x: Integer::new(), y:0}],
        }
    }

    pub fn add_point(&mut self, x: Integer, y: i64) -> Result<bool> {
        if self.points_in_hull > 0 && x <= self.last_x {
            return Err(anyhow!("point does not increase"));
        }

        self.last_x = x.clone();
        self.last_y = y;
        let p1 = Point {
            x: x.clone(),
            y: y + self.epsilon,
        };
        let p2 = Point {
            x: x.clone(),
            y: y - self.epsilon,
        };

        if self.points_in_hull == 0 {
            self.first_x = x.clone();
            self.first_y = y;
            self.rectangle[0] = p1.clone();
            self.rectangle[1] = p2.clone();
            self.upper.clear();
            self.lower.clear();
            self.upper.push(p1);
            self.lower.push(p2);
            self.upper_start = 0;
            self.lower_start = 0;
            self.points_in_hull += 1;
            return Ok(true);
        }

        if self.points_in_hull == 1 {
            self.rectangle[2] = p2.clone();
            self.rectangle[3] = p1.clone();
            self.upper.push(p1);
            self.lower.push(p2);
            self.points_in_hull += 1;
            return Ok(true);
        }

        let slope1 = &self.rectangle[2] - &self.rectangle[0];
        let slope2 = &self.rectangle[3] - &self.rectangle[1];
        let outside_line1 = &p1 - &self.rectangle[2] < slope1;
        let outside_line2 = &p2 - &self.rectangle[3] > slope2;

        if outside_line1 || outside_line2 {
            self.points_in_hull = 0;
            self.last_y -= 1;
            return Ok(false);
        }

        if &p1 - &self.rectangle[1] < slope2 {
            //Find extreme slope
            let mut min = &self.lower[self.lower_start] - &p1;
            let mut min_i = self.lower_start;
            let mut i = self.lower_start + 1;
            while i < self.lower.len() {
                let val = &self.lower[i] - &p1;
                if val > min {
                    break;
                }
                min = val;
                min_i = i;
                i += 1;
            }

            self.rectangle[1] = self.lower[min_i].clone();
            self.rectangle[3] = p1.clone();
            self.lower_start = min_i;

            // Hull update
            let mut end = self.upper.len();
            while end >= self.upper_start + 2 && cross(&self.upper[end-2], &self.upper[end-1], &p1) <= 0 {
                end -= 1;
            }
            
            self.upper = self.upper[0..end].to_vec();
            self.upper.push(p1);
        }

        if &p2 - &self.rectangle[0] > slope1 {
            // Find extreme slope
            let mut max = &self.upper[self.upper_start] - &p2;
            let mut max_i = self.upper_start;
            let mut i = self.upper_start + 1;
            while i < self.upper.len() {
                let val = &self.upper[i] - &p2;
                if val < max {
                    break;
                }
                max = val;
                max_i = i;
                i += 1;
            }

            self.rectangle[0] = self.upper[max_i].clone();
            self.rectangle[2] = p2.clone();
            self.upper_start = max_i;

            // Hull update
            let mut end = self.lower.len();
            while end >= self.lower_start + 2 && cross(&self.lower[end-2], &self.lower[end-1], &p2) >= 0 {
                end -= 1;
            }
            
            self.lower = self.lower[0..end].to_vec();
            self.lower.push(p2);
        }

        self.points_in_hull += 1;
        Ok(true)
    }

    pub fn reset(&mut self) {
        self.points_in_hull = 0;
        self.lower.clear();
        self.upper.clear();
    }

    pub fn get_segment(&self) -> CanonicalSegment {
        if self.points_in_hull == 1 {
            return CanonicalSegment::new(&self.rectangle[0], &self.rectangle[1], self.last_y);
        }
        return CanonicalSegment::new_long(&self.rectangle, self.last_y);
    }

    pub fn get_points_in_hull(&self) -> usize {
        self.points_in_hull
    }
}

#[derive(Debug, Clone)]
pub struct CanonicalSegment {
    rectangle: [Point; 4],
    last_y: i64,
}

impl CanonicalSegment {
    pub fn new(p0: &Point, p1: &Point, last_y: i64) -> Self {
        Self {
            rectangle: [p0.clone(), p1.clone(), p0.clone(), p1.clone()],
            last_y,
        }
    }

    pub fn new_long(rec: &[Point;4], last_y: i64) -> Self {
        Self {
            rectangle: [rec[0].clone(), rec[1].clone(), rec[2].clone(), rec[3].clone()],
            last_y,
        }
    }

    pub fn one_point(&self) -> bool {
        self.rectangle[0].x == self.rectangle[2].x && self.rectangle[0].y == self.rectangle[2].y
        && self.rectangle[1].x == self.rectangle[3].x && self.rectangle[1].y == self.rectangle[3].y
    }

    pub fn get_last_y(&self) -> i64 {
        self.last_y
    }

    pub fn get_intersection(&self) -> (Rational, Rational){
        let p0 = &self.rectangle[0];
        let p1 = &self.rectangle[1];
        let p2 = &self.rectangle[2];
        let p3 = &self.rectangle[3];
        let slope1 = p2 - p0;
        let slope2 = p3 - p1;

        if self.one_point() || slope1 == slope2 {
            return (Rational::from(p0.x.clone()), Rational::from(p0.y));
        }

        let p0p1 = p1 - p0;
        let a = (&slope1.dx * slope2.dy).complete() - (slope1.dy * &slope2.dx).complete();
        // let b = (p0p1.dx * slope2.dy - p0p1.dy * slope2.dx) / a;
        let b = Rational::from((p0p1.dx * slope2.dy - p0p1.dy * slope2.dx, a));
        let i_x = &p0.x + (&b * &slope1.dx).complete();
        let i_y = p0.y + b * slope1.dy;
        return (i_x, i_y);
    }

    pub fn get_floating_point_segment(&self, origin: Integer) -> (Rational, Rational) {
        if self.one_point() {
            let left = Rational::new();
            let right = Rational::from((self.rectangle[0].y + self.rectangle[1].y , 2));
            return (left, right);
        }

        // here we assume that both numbers are float

        let (i_x, i_y) = self.get_intersection();
        let (min_slope, max_slope) = self.get_slope_range();
        let slope = (min_slope + max_slope) / Rational::from(2);
        let intercept = i_y - (i_x - origin) * &slope;
        return (slope, intercept);
    }

    pub fn get_slope_range(&self) -> (Rational, Rational){
        if self.one_point() {
            return (Rational::new(), Rational::from(1));
        }

        let min_slope = &self.rectangle[2] - &self.rectangle[0];
        let max_slope = &self.rectangle[3] - &self.rectangle[1];
        
        return (Rational::from((min_slope.dy, min_slope.dx)), Rational::from((max_slope.dy, max_slope.dx)));
    }
}

pub fn compound_key_to_integer(key: &CompoundKey) -> Integer {
    let bytes = key.to_bytes();
    let i = Integer::from_digits(&bytes, Order::Msf);
    return i;
}

#[derive(Debug, Clone)]
pub struct ModelGenerator {
    pub start: Option<CompoundKey>,
    pub pgm: OptimalPiecewiseLinearModel,
}

impl ModelGenerator {
    pub fn new(epsilon: i64) -> Self {
        Self {
            start: None,
            pgm: OptimalPiecewiseLinearModel::new(epsilon),
        }
    }

    pub fn append(&mut self, key: &CompoundKey, pos: usize) -> bool {
        // set the start key of the model
        if self.start.is_none() {
            self.start = Some(*key);
        }
        // transform the compound key to the corresponding big integer
        let key_int = compound_key_to_integer(key);
        // handling case of the same keys
        let last_point_in_pgm = &self.pgm.last_x;
        if &key_int == last_point_in_pgm && pos as i64 == self.pgm.last_y {
            return true;
        }
        // try to insert the point to the convex hull
        let r = self.pgm.add_point(key_int, pos as i64).unwrap();
        return r;
    }

    pub fn finalize_model(&mut self) -> CompoundKeyModel {
        let seg = self.pgm.get_segment();
        let start = self.start.unwrap().clone();
        let start_int = compound_key_to_integer(&start);
        let last_pos = seg.get_last_y();
        let (slope, intercept) = seg.get_floating_point_segment(start_int);
        self.pgm.reset();
        self.start = None;
        CompoundKeyModel {
            start,
            slope: slope.to_f64(),
            intercept: intercept.to_f64(),
            last_index: last_pos as u32,
        }
    }

    pub fn is_hull_empty(&self) -> bool {
        if self.pgm.get_points_in_hull() != 0 {
            return false;
        } else {
            return true;
        }
    }
}
pub const MODEL_SIZE: usize = COMPOUND_KEY_SIZE + 8 + 8 + 4;
#[derive(Debug, Default, Clone, Copy)]
pub struct CompoundKeyModel {
    pub start: CompoundKey,
    pub slope: f64,
    pub intercept: f64,
    pub last_index: u32,
}

impl PartialEq for CompoundKeyModel {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && (self.slope - other.slope) < 1.0e-6 && (self.intercept - other.intercept) < 1.0e-6 && self.last_index == other.last_index
    }

    fn ne(&self, other: &Self) -> bool {
        self.start != other.start || (self.slope - other.slope) > 1.0e-6 || (self.intercept - other.intercept) > 1.0e-6 || self.last_index != other.last_index
    }
}

impl CompoundKeyModel {
    pub fn get_slope_intercept(&self) -> (f64, f64) {
        (self.slope, self.intercept)
    }

    pub fn get_start(&self) -> CompoundKey {
        self.start.clone()
    }

    pub fn get_last_index(&self) -> u32 {
        self.last_index
    }

    pub fn predict(&self, key: &Integer) -> usize {
        let start_int = compound_key_to_integer(&self.start);
        let pos = ((key - start_int) * &Rational::from_f64(self.slope).unwrap()).complete() + &Rational::from_f64(self.intercept).unwrap();
        let pos_integer = pos.to_f64().floor() as usize;
        let max_index = self.last_index as usize;
        if pos_integer > max_index {
            return max_index;
        } else {
            return pos_integer;
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut total_bytes = self.start.to_bytes();
        total_bytes.extend(&self.slope.to_be_bytes());
        total_bytes.extend(&self.intercept.to_be_bytes());
        total_bytes.extend(&self.last_index.to_be_bytes());
        return total_bytes;
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let start = CompoundKey::from_bytes(&bytes[0..56]);
        let slope = f64::from_be_bytes(bytes[56..64].try_into().expect("err"));
        let intercept = f64::from_be_bytes(bytes[64..72].try_into().expect("err"));
        let last_index = u32::from_be_bytes(bytes[72..76].try_into().expect("err"));
        Self { 
            start,
            slope,
            intercept,
            last_index,
        }
    }
}

pub fn fetch_model_and_predict(final_models: &Vec<CompoundKeyModel>, key: &CompoundKey) -> usize {
    let key_integer = compound_key_to_integer(&key);
    let mut index;
    let model_len = final_models.len();
    let mut l: i32 = 0;
    let mut r: i32 = model_len as i32 - 1;
    // println!("model len {:?}", r);
    if model_len == 0 {
        return final_models[0].predict(&key_integer);
    }

    while l <= r && l >=0 && r <= model_len as i32 - 1{
        let m = l + (r - l) / 2;
        if final_models[m as usize].start < *key {
            l = m + 1;
        }
        else if final_models[m as usize].start > *key {
            r = m - 1;
        }
        else {
            index = m as usize;
            return final_models[index].predict(&key_integer);
        }
    }

    index = l as usize;

    if index == model_len {
        index -= 1;
    }

    if *key < final_models[index].start && index > 0 {
        index -= 1;
    }
    final_models[index].predict(&key_integer)
}
#[cfg(test)]
mod tests {
    use rug::Integer;
    use rand::prelude::*;
    use crate::types::{CompoundKey, AddrKey};
    use crate::{H160, H256};
    use super::{ModelGenerator, CompoundKeyModel, fetch_model_and_predict};
    use crate::{OpenOptions};
    use std::io::{Write, self, Seek, Read, SeekFrom};
    #[test]
    fn test_big_number() {
        let a = Integer::from(10);
        let b = 5 - a;
        assert_eq!(b, 5 - 10);
    }

    #[test]
    fn test_compound_key() {
        let epsilon = 23;
        let num_of_addr = 2;
        let num_of_contract = 2;
        let num_of_version = 30;
        let mut rng = StdRng::seed_from_u64(1);
        let mut addr_key_vec = Vec::<AddrKey>::new();
        let mut keys = Vec::<CompoundKey>::new();
        for _ in 1..=num_of_contract {
            for _ in 1..=num_of_addr {
                let acc_addr = H160::random_using(&mut rng);
                let state_addr = H256::random_using(&mut rng);
                let addr_key = AddrKey::new(acc_addr.into(), state_addr.into());
                addr_key_vec.push(addr_key);
            }
        }
        for addr_key in &addr_key_vec {
            for k in 1..=num_of_version {
                let compound_key = CompoundKey::new_with_addr_key(*addr_key, k * 2);
                keys.push(compound_key);
            }
        }
        
        // keys.push(CompoundKey::new(H160::from_low_u64_be(0), H256::from_low_u64_be(0), 0));
        keys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        println!("complete sort");
        let mut model_generator = ModelGenerator::new(epsilon);
        let mut model_vec: Vec<CompoundKeyModel> = Vec::new();
        for (i, key) in keys.iter().enumerate() {
            let r = model_generator.append(&key, i);
            if r == false {
                let model = model_generator.finalize_model();
                model_vec.push(model);
                model_generator.append(&key, i);
            }
        }

        if !model_generator.is_hull_empty() {
            let model = model_generator.finalize_model();
            model_vec.push(model);
        }

        println!("len of model_vec: {}", model_vec.len());
        println!("model: {:?}", model_vec);
        for (i, key) in keys.iter().enumerate() {
            let predict_pos = fetch_model_and_predict(&model_vec, key);
            let diff = (i as f64 - predict_pos as f64).abs().floor();
            if diff > (epsilon + 1) as f64 {
                println!("i: {}, pred_pos: {}, diff: {}", i, predict_pos, diff);
            }
            // assert!((i as f64 - predict_pos as f64).abs().floor() <= (epsilon + 1) as f64);
        }
        let low = CompoundKey::new_with_addr_key(keys[0].addr, 0);
        println!("low: {:?}", low);
        let pred = fetch_model_and_predict(&model_vec, &low);
        println!("pred: {}", pred);
    }
    #[test]
    fn test_pgm() {
        let epsilon = 2;
        let n = 100000;
        let mut rng = StdRng::seed_from_u64(1);
        let mut keys = Vec::<CompoundKey>::new();
        for i in 0..n {
            let acc_addr = H160::random_using(&mut rng);
            let state_addr = H256::random_using(&mut rng);
            let version = i as u32;
            let key = CompoundKey::new(acc_addr.into(), state_addr.into(), version);
            keys.push(key);
        }

        // let num_of_contract = 10;
        // let num_of_address = 10;
        // let num_of_versions = 1;
        // let mut addr_vec = Vec::<AddrKey>::new();

        // let mut keys = Vec::<CompoundKey>::new();
        // let mut rng = StdRng::seed_from_u64(1);
        // for _ in 1..=num_of_contract {
        //     for _ in 1..=num_of_address {
        //         let contract_addr = H160::random_using(&mut rng);
        //         let state_addr = H256::random_using(&mut rng);
        //         let compound_key = AddrKey::new(contract_addr, state_addr);
        //         addr_vec.push(compound_key);
        //     }
        // }

        // let multi_factor = 1;
        // for addr in &addr_vec {
        //     for k in 1..=num_of_versions {
        //         let compound_key = CompoundKey::new(addr.acc_addr, addr.state_addr, k * multi_factor);
        //         keys.push(compound_key);
        //     }
        // }

        keys.sort_by(|a, b| a.partial_cmp(b).unwrap());

        println!("complete sort");
        let mut model_generator = ModelGenerator::new(epsilon);
        let mut model_vec: Vec<CompoundKeyModel> = Vec::new();
        for (i, key) in keys.iter().enumerate() {
            let r = model_generator.append(&key, i);
            if r == false {
                let model = model_generator.finalize_model();
                model_vec.push(model);
                model_generator.append(&key, i);
            }
        }

        if !model_generator.is_hull_empty() {
            let model = model_generator.finalize_model();
            model_vec.push(model);
        }

        println!("len of model_vec: {}", model_vec.len());

        for (i, key) in keys.iter().enumerate() {
            let predict_pos = fetch_model_and_predict(&model_vec, key);
            let diff = (i as f64 - predict_pos as f64).abs().floor();
            if diff > (epsilon + 1) as f64 {
                println!("i: {}, pred_pos: {}, diff: {}", i, predict_pos, diff);
            }
            // assert!((i as f64 - predict_pos as f64).abs().floor() <= (epsilon + 1) as f64);
        }
    }

    #[test]
    fn test_io_copy() {
        let mut file1 = OpenOptions::new().create(true).read(true).write(true).truncate(true).open("file1.dat").unwrap();
        let mut file2 = OpenOptions::new().create(true).read(true).write(true).truncate(true).open("file2.dat").unwrap();
        let n = 100000;
        for _ in 0..n {
            file1.write_all(&[1u8; 4096]).unwrap();
            file2.write_all(&[2u8; 4096]).unwrap();
        }
        file1.rewind().unwrap();
        println!("sleep 10 s");
        std::thread::sleep(std::time::Duration::from_secs(30));
        let r = io::copy(&mut file1, &mut file2).unwrap();
        println!("{}", r);
        println!("sleep 10 s");
        std::thread::sleep(std::time::Duration::from_secs(30));
    }
    use std::thread;
    #[test]
    fn test_read_one_file() {
        let n = 100;
        
        let handle = thread::spawn(move || {
            let mut file2 = OpenOptions::new().read(true).open("state.dat").unwrap();
            for i in 0..n {
                let mut bytes = [0u8; 4096];
                file2.seek(SeekFrom::Start(i as u64)).unwrap();
                file2.read_exact(&mut bytes).unwrap();
                println!("file2: {:?}", bytes.len());
            }
        });
        let mut file1 = OpenOptions::new().read(true).open("state.dat").unwrap();
        for i in 0..n {
            let mut bytes = [0u8; 4096];
            file1.seek(SeekFrom::Start(i as u64)).unwrap();
            file1.read_exact(&mut bytes).unwrap();
            println!("file1: {:?}", bytes.len());
        }
        
        handle.join().unwrap();
    }
}