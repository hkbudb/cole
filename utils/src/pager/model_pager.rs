use crate::{File, OpenOptions, models::CompoundKeyModel, types::CompoundKey, cacher::CacheManager};
use std::io::{Seek, SeekFrom};
use super::{Page, PAGE_SIZE, MAX_NUM_MODEL_IN_PAGE};
use std::os::unix::prelude::FileExt;
use crate::models::{ModelGenerator, fetch_model_and_predict};
use std::cmp::{max, min};
pub const TMP_MODEL_FILE_NAME: &str = "tmp_model_file.dat";
/* A helper structure to keep a collection of models and their sharing model_level
 */
#[derive(Debug, Clone)]
pub struct ModelCollections {
    pub v: Vec<CompoundKeyModel>, // vector of models
    pub model_level: u32, 
}

impl ModelCollections {
    pub fn new() -> Self {
        Self {
            v: vec![],
            model_level: 0,
        }
    }

    pub fn init_with_model_level(model_level: u32) -> Self {
        Self {
            v: vec![],
            model_level,
        }
    }
}

/* A helper that writes the models into a file with a sequence of pages, similar to the StatePageWriter
 */
pub struct ModelPageWriter {
    pub file: File, // file object of the corresponding index file
    pub latest_model_collection: ModelCollections, // a preparation vector to obsorb the streaming models which are not persisted in the file yet
    pub num_stored_pages: usize, // records the number of pages that are stored in the file
}

impl ModelPageWriter {
    /* Initialize the writer using a given file_name
     */
    pub fn create(file_name: &str, model_level: u32) -> Self {
        let file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(&file_name).unwrap();
        Self {
            file,
            latest_model_collection: ModelCollections::init_with_model_level(model_level),
            num_stored_pages: 0,
        }
    }

    /* Load the writer from a given file, cache is empty, num_states and num_stored_pages are derived from the file
     */
    pub fn load(file_name: &str) -> Self {
        let file = OpenOptions::new().create(true).read(true).write(true).open(&file_name).unwrap();
        let mut num_stored_pages = file.metadata().unwrap().len() as usize / PAGE_SIZE;
        let mut latest_model_collection = ModelCollections::new();
        if num_stored_pages > 0 {
            let last_page_offset = (num_stored_pages - 1) * PAGE_SIZE;
            // get last page from file
            let mut bytes = [0u8; PAGE_SIZE];
            file.read_exact_at(&mut bytes, last_page_offset as u64).unwrap();
            let page = Page::from_array(bytes);
            let r = page.to_model_vec();
            let page_vec = r.v;
            latest_model_collection.model_level = r.model_level;
            // derive number of models in the last page
            let num_states_in_last_page = page_vec.len();

            if num_states_in_last_page != MAX_NUM_MODEL_IN_PAGE {
                // the last page is not full, should not be finalized in the file
                latest_model_collection.v = page_vec;
                num_stored_pages -= 1;
            }
        }
        Self {
            file,
            latest_model_collection,
            num_stored_pages,
        }
    }

    /* Streamingly add the model to the latest collection, if the collection is full, flush it to the file
     */
    pub fn append(&mut self, model: CompoundKeyModel) {
        // add the model
        self.latest_model_collection.v.push(model);
        if self.latest_model_collection.v.len() == MAX_NUM_MODEL_IN_PAGE {
            // vector is full, should be added to a page and flushed the page to the file
            self.flush();
        }
    }
    /* Flush the vector in latest update page to the last page in the value file
     */
    pub fn flush(&mut self) {
        if self.latest_model_collection.v.len() != 0 {
            // first put the vector into a page
            let page = Page::from_model_vec(&self.latest_model_collection.v, self.latest_model_collection.model_level);
            // compute the offset at which the page will be written in the file
            let offset = self.num_stored_pages * PAGE_SIZE;
            // write the page to the file
            self.file.write_all_at(&page.block, offset as u64).unwrap();
            // clear the vector
            self.latest_model_collection.v.clear();
            self.num_stored_pages += 1;
        }
    }

    /* Transform writer to reader
     */
    pub fn to_model_reader(self) -> ModelPageReader {
        let file = self.file;
        let num_stored_pages = self.num_stored_pages;
        ModelPageReader {
            file,
            num_stored_pages,
        }
    }
}

/* A helper to read the models from the file
 */
pub struct ModelPageReader {
    pub file: File, // file object of the corresponding index file
    pub num_stored_pages: usize, // records the number of pages that are stored in the file
}

impl ModelPageReader {
    /* Load the reader from a given file
     cache is empty and num_stored_pages are derived from the file
     */
    pub fn load(file_name: &str) -> Self {
        let file = OpenOptions::new().create(true).read(true).write(true).open(&file_name).unwrap();
        let num_stored_pages = file.metadata().unwrap().len() as usize / PAGE_SIZE;
        Self {
            file,
            num_stored_pages,
        }
    }

    /* Load the deserialized vector of the page from the file at given page_id
     */
    pub fn read_deser_page_at(&mut self, run_id: u32, page_id: usize, cache_manager: &mut CacheManager) -> ModelCollections {
        // first check whether the cache contains the page
        let r = cache_manager.read_model_cache(run_id, page_id);
        if r.is_some() {
            // cache contains the page
            let v = r.unwrap().clone();
            return v;
        } else {
            // cache does not contain the page, should load the page from the file
            let offset = page_id * PAGE_SIZE;
            let mut bytes = [0u8; PAGE_SIZE];
            self.file.read_exact_at(&mut bytes, offset as u64).unwrap();
            let page = Page::from_array(bytes);
            let v = page.to_model_vec();
            // before return the vector, add it to the cache with page_id as the key
            cache_manager.set_model_cache(run_id, page_id, v.clone());
            return v;
        }
    }

    /* Query Models in the model file
     */
    pub fn get_pred_state_pos(&mut self, run_id: u32, search_key: &CompoundKey, epsilon: i64, cache_manager: &mut CacheManager) -> usize {
        /* First load the last page and find the model that covers the search key
         */
        let last_page_id = self.num_stored_pages - 1;
        let top_model_collection = self.read_deser_page_at(run_id, last_page_id, cache_manager);
        let (model_v, mut model_level) = (top_model_collection.v, top_model_collection.model_level);
        if model_level == 0 {
            // the last page stores the lowest level
            let pred_pos = fetch_model_and_predict(&model_v, search_key);
            return pred_pos;
        } else {
            // first search the model in the model_v and then determine the predicted page id range
            let mut pred_pos = fetch_model_and_predict(&model_v, search_key);
            let pred_page_id_lb = max(0, (pred_pos as i64 - epsilon - 1) / MAX_NUM_MODEL_IN_PAGE as i64) as usize;
            let pred_page_id_ub = min(last_page_id, (pred_pos + epsilon as usize + 1) / MAX_NUM_MODEL_IN_PAGE);
            model_level -= 1;
            pred_pos = self.query_model(run_id, pred_page_id_lb, pred_page_id_ub, search_key, model_level, cache_manager);
            while model_level != 0 {
                let pred_page_id_lb = max(0, (pred_pos as i64 - epsilon -1) / MAX_NUM_MODEL_IN_PAGE as i64) as usize;
                let pred_page_id_ub = min(last_page_id, (pred_pos + epsilon as usize + 1) / MAX_NUM_MODEL_IN_PAGE);
                model_level -= 1;
                pred_pos = self.query_model(run_id, pred_page_id_lb, pred_page_id_ub, search_key, model_level, cache_manager);
            }
            return pred_pos;
        }
    }

    fn query_model(&mut self, run_id: u32, page_id_lb: usize, page_id_ub: usize, search_key: &CompoundKey, model_level: u32, cache_manager: &mut CacheManager) -> usize {
        let mut model_v = Vec::<CompoundKeyModel>::new();
        for page_id in page_id_lb ..= page_id_ub {
            let collection = self.read_deser_page_at(run_id, page_id, cache_manager);
            if collection.model_level == model_level {
                model_v.extend(&collection.v);
            }
        }
/*         for (i, model) in model_v.iter().enumerate() {
            println!("i: {}, model: {:?}", i, model);
        } */
        let pred_pos = fetch_model_and_predict(&model_v, search_key);
        return pred_pos;
    }
}

/* A model constructor that generates and appends models to the file in a streaming fashion
 */
pub struct StreamModelConstructor {
    pub output_model_writer: ModelPageWriter, // a writer of the output model file
    pub lowest_level_model_generator: ModelGenerator, // a model generator of the lowest level (learn input is the states)
    pub epsilon: i64, // an upper-bound model prediction error
    pub state_pos: usize, // the position of the input state
}

impl StreamModelConstructor {
    /* Initiate the constructor with the output model file name and the upper error bound
     */
    pub fn new(output_file_name: &str, epsilon: i64) -> Self {
        // create the output model writer
        let output_model_writer = ModelPageWriter::create(output_file_name, 0);
        // initiate the model generator for the lowest level
        let lowest_level_model_generator = ModelGenerator::new(epsilon);
        Self {
            output_model_writer,
            lowest_level_model_generator,
            epsilon,
            state_pos: 0,
        }
    }

    /* Streaminly append the key to the model generator for the lowest level
     */
    pub fn append_state_key(&mut self, key: &CompoundKey) {
        let pos = self.state_pos;
        let r = self.lowest_level_model_generator.append(key, pos);
        if r == false {
            // finalize the model since the new coming key cannot fit the model within the prediction error bound
            let model = self.lowest_level_model_generator.finalize_model();
            // write the model to the output model writer (can be kept in the latest page cache in memory or be flushed to the file)
            self.output_model_writer.append(model);
            // re-insert the key position to the model generator since the previous insertion fails
            self.lowest_level_model_generator.append(key, pos);
        }
        self.state_pos += 1;
    }

    /* Finalize the append of the key-pos
       End the insertion of the lowest_level_model_generator: finalize the model, append it to the output_model_writer, and flush it to the file
       Recursively build the models upon the previous level and append them to the file in a streaming fashion.
     */
    pub fn finalize_append(&mut self) {
        /* First finalize the lowest level models
         */
        if !self.lowest_level_model_generator.is_hull_empty() {
            let model = self.lowest_level_model_generator.finalize_model();
            self.output_model_writer.append(model);
        }
        self.output_model_writer.flush();

        /*
        recursively construct models in the upper levels
        */
        
        let output_model_writer = &mut self.output_model_writer;
        // n is the number of page in the previous model level
        let mut n = output_model_writer.num_stored_pages;
        let mut model_level = 0;
        while n > 1 {
            // n > 1 means we should build an upper level models since the top level models should be kept in a single page
            // increment the model_level
            model_level += 1;
            // start_page_id is the id of the starting page that the upper level is learned from
            let start_page_id = output_model_writer.num_stored_pages - n;
            // initiate a model generator for the upper level models
            let mut model_generator = ModelGenerator::new(self.epsilon);
            // create a temporary model writer for keeping the upper level models
            let mut tmp_model_writer = ModelPageWriter::create(TMP_MODEL_FILE_NAME, model_level);
            // pos is the position of the learned input of the upper level models
            let mut pos = start_page_id * MAX_NUM_MODEL_IN_PAGE;
            
            for page_id in start_page_id .. output_model_writer.num_stored_pages {
                // read the model page from the file in the output_model_writer at the corresponding offset
                let offset = page_id * PAGE_SIZE;
                let mut bytes = [0u8; PAGE_SIZE];
                output_model_writer.file.read_exact_at(&mut bytes, offset as u64).unwrap();
                let page = Page::from_array(bytes);
                // deserialize the models from the page, these are seen as the learned input of the upper level models
                let input_models = page.to_model_vec().v;
                for input_model in input_models {
                    let r = model_generator.append(&input_model.start, pos);
                    if r == false {
                        let output_model = model_generator.finalize_model();
                        // write the output model to the temporary model write of the upper models
                        tmp_model_writer.append(output_model);
                    }
                    model_generator.append(&input_model.start, pos);
                    pos += 1;
                }
            }
            
            // handle the rest of the points in the hull
            if !model_generator.is_hull_empty() {
                let output_model = model_generator.finalize_model();
                tmp_model_writer.append(output_model);
            }
            // flush the temporary model writer to the temporary file
            tmp_model_writer.flush();
            // update n as the number of page of the temporary model writer
            n = tmp_model_writer.num_stored_pages;
            // concatenate the content of temporary model file to the output model file
            concatenate_file_a_to_file_b(&mut tmp_model_writer.file, &mut output_model_writer.file);
            // update the number of pages in the output model file
            output_model_writer.num_stored_pages += tmp_model_writer.num_stored_pages;
            
            // drop the tmp_model_writer and remove the temporary file
            drop(tmp_model_writer);
            std::fs::remove_file(TMP_MODEL_FILE_NAME).unwrap();
        }
    }
}

pub fn concatenate_file_a_to_file_b(file_a: &mut File, file_b: &mut File) {
    // rewind the cursor of file_a to the start
    file_a.rewind().unwrap();
    let l = file_b.metadata().unwrap().len();
    // seek the cursor of file_b to the end
    file_b.seek(SeekFrom::Start(l)).unwrap();
    // copy the content in file_a to the end of file_b
    std::io::copy(file_a, file_b).unwrap();
}

#[cfg(test)]
mod tests {
    use crate::{H160, H256};
    use rand::{rngs::StdRng, SeedableRng};
    use crate::types::CompoundKey;

    use super::*;

    #[test]
    fn test_model_pager() {
        let n = 10000;
        let mut writer = ModelPageWriter::create("model.dat", 0);
        let mut model_vec = Vec::<CompoundKeyModel>::new();
        for i in 0..n {
            let acc_addr = H160::random();
            let state_addr = H256::random();
            let version = i as u32;
            let key = CompoundKey::new(acc_addr.into(), state_addr.into(), version);
            let model = CompoundKeyModel {
                start: key,
                slope: 1.0,
                intercept: 2.0,
                last_index: i as u32,
            };
            model_vec.push(model.clone());
            writer.append(model);
        }
        writer.flush();
        let mut reader = writer.to_model_reader();
        let mut cache_manager = CacheManager::new();
        for j in 0..5 {
            // iteratively read the pages
            let start = std::time::Instant::now();
            for i in 0..n {
                let page_id = i / MAX_NUM_MODEL_IN_PAGE;
                let inner_page_pos = i % MAX_NUM_MODEL_IN_PAGE;
                let collections = reader.read_deser_page_at(0, page_id, &mut cache_manager);
                let state = collections.v[inner_page_pos];
                assert_eq!(state, model_vec[i]);
            }
            let elapse = start.elapsed().as_nanos() as usize / n;
            println!("round {}, read state time: {}", j, elapse);
        }
    }

    #[test]
    fn test_streaming_model() {
        let mut rng = StdRng::seed_from_u64(1);
        let epsilon = 46;
        let n = 1000000;
        
        let mut keys = Vec::<CompoundKey>::new();
        for i in 0..n {
            let acc_addr = H160::random_using(&mut rng);
            let state_addr = H256::random_using(&mut rng);
            let version = i as u32;
            let key = CompoundKey::new(acc_addr.into(), state_addr.into(), version);
            keys.push(key);
        }
        keys.sort();
        let start = std::time::Instant::now();
        let mut stream_model_constructor = StreamModelConstructor::new("model.dat", epsilon);
        let mut point_vec = Vec::<(CompoundKey, usize)>::new();
        for (i, key) in keys.iter().enumerate() {
            stream_model_constructor.append_state_key(key);
            point_vec.push((*key, i));
        }
        stream_model_constructor.finalize_append();
        let elapse = start.elapsed().as_nanos();
        println!("avg construct time: {}", elapse / n as u128);
        let writer = stream_model_constructor.output_model_writer;
        let mut reader = writer.to_model_reader();
        let mut cache_manager = CacheManager::new();
        let num_of_pages = reader.num_stored_pages;
        for i in 0..num_of_pages {
            let collection = reader.read_deser_page_at(0, i, &mut cache_manager);
            println!("collection size: {:?}, model_level: {:?}", collection.v.len(), collection.model_level);
        }

        let start = std::time::Instant::now();
        for point in point_vec {
            let key = point.0;
            let true_pos = point.1;
            let pred_pos = reader.get_pred_state_pos(0, &key, epsilon, &mut cache_manager);
            if (true_pos as f64 - pred_pos as f64).abs() > (epsilon + 1) as f64 {
                println!("true_pos: {}, pred_pos: {}, diff: {}", true_pos, pred_pos, (true_pos as f64 - pred_pos as f64).abs());
            }
            // println!("true: {}, pred: {}, diff: {}", true_pos, pred_pos, (true_pos as f64 - pred_pos as f64).abs());
            // assert!((true_pos as f64 - pred_pos as f64).abs().floor() <= (epsilon + 1) as f64);
        }
        let elapse = start.elapsed().as_nanos();
        println!("avg pred time: {}", elapse / n as u128);
        std::thread::sleep(std::time::Duration::from_secs(30));
    }
}