package task1;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.desc;
import static org.apache.spark.sql.functions.explode;
import static org.apache.spark.sql.functions.row_number;
import static org.apache.spark.sql.functions.split;

/**
 * Task 1 main class contains methods to work with imdb data
 *
 * Data can be found by link https://datasets.imdbws.com
 * Unpacked tsv files is located in resources/task1/input folder
 *
 * Each method saves output to resources/task1/output folder
 */
public class IMDbTask {
    private static final String BASICS_FILE = "src/main/resources/task1/input/title.basics.tsv";
    private static final String RATINGS_FILE = "src/main/resources/task1/input/title.ratings.tsv";
    private static final String PRINCIPALS_FILE = "src/main/resources/task1/input/title.principals.tsv";
    private static final String NAMES_FILE = "src/main/resources/task1/input/name.basics.tsv";

    private static final SparkSession spark = SparkSession.builder()
        .appName("task1").master("local[*]").getOrCreate();

    public static void main(String[] args) {
        topAll();
        topByGenre();
        topByGenreAndDecade();
        topActors();
        topDirectorFilms();
        spark.close();
    }

    /**
     * Find top 100 films with more than 100 000 votes
     */
    private static void topAll() {
        Dataset<Row> basDs = loadDataset(BASICS_FILE);
        Dataset<Row> ratDs = loadDataset(RATINGS_FILE);

        basDs = basDs.join(ratDs)
            .where(basDs.col("tconst").equalTo(ratDs.col("tconst")))
            .select(basDs.col("tconst"), col("primaryTitle"), col("numVotes"),
                col("averageRating"), col("startYear"), col("titleType"));

        basDs = basDs
            .select("*")
            .where(col("titleType").equalTo("movie")
                .and(col("numVotes").cast(DataTypes.IntegerType).geq(100000)))
            .sort(desc("averageRating"))
            .limit(100)
            .drop("titleType");

        saveToFile("src/main/resources/task1/output/top-all", basDs);

    }

    /**
     * Find top 10 films of each genre
     */
    private static void topByGenre() {
        Dataset<Row> basDs = loadDataset(BASICS_FILE);
        Dataset<Row> ratDs = loadDataset(RATINGS_FILE);

        basDs = basDs.join(ratDs)
            .where(basDs.col("tconst").equalTo(ratDs.col("tconst")))
            .select(basDs.col("tconst"), col("primaryTitle"), col("numVotes"),
                col("averageRating"), col("startYear"), col("titleType"),
                col("genres"))
            .where(col("titleType").equalTo("movie")
                .and(col("numVotes").cast(DataTypes.IntegerType).geq(100000)))
            .withColumn("genre", explode(split(col("genres"), "\\s*,\\s*")))
            .drop("genres");

        WindowSpec windowsSpec = Window.partitionBy(col("genre"))
            .orderBy(desc("averageRating"), desc("numVotes"));

        basDs = basDs.withColumn("row_number", row_number().over(windowsSpec))
            .where(col("row_number").leq(10))
            .drop(col("row_number"));

        saveToFile("src/main/resources/task1/output/top-by-genre", basDs);
    }

    /**
     * Find top 10 films of each genre per each decade from 1950
     */
    private static void topByGenreAndDecade() {
        Dataset<Row> basDs = loadDataset(BASICS_FILE);
        Dataset<Row> ratDs = loadDataset(RATINGS_FILE);
        spark.udf().register("toDecade", (String year) -> {
            int decade = (int) (Math.floor(Integer.parseInt(year)/10.0)*10);
            return decade + " - " + (decade + 10);
        }, DataTypes.StringType);

        basDs = basDs.join(ratDs)
            .where(basDs.col("tconst").equalTo(ratDs.col("tconst"))
                .and(col("startYear").geq(1950)))
            .select(basDs.col("tconst"), col("primaryTitle"), col("numVotes"),
                col("averageRating"), col("startYear"), col("titleType"),
                col("genres"))
            .where(col("titleType").equalTo("movie")
                .and(col("numVotes").cast(DataTypes.IntegerType).geq(100000)))
            .withColumn("genre", explode(split(col("genres"), "\\s*,\\s*")))
            .withColumn("yearRange", callUDF("toDecade", col("startYear")))
            .drop("genres", "titleType");

        WindowSpec windowsSpecGenre = Window.partitionBy(col("yearRange"), col("genre"))
            .orderBy(desc("averageRating"), desc("numVotes"));

        basDs = basDs
            .withColumn("row_number", row_number().over(windowsSpecGenre))
            .where(col("row_number").leq(10))
            .orderBy(desc("yearRange"), desc("genre"))
            .drop(col("row_number"));

        saveToFile("src/main/resources/task1/output/top-by-genre-decade", basDs);
    }

    /**
     * Find all actors who starred in popular films 10 or more times
     * Popular film is film with more than 100 000 votes
     */
    private static void topActors() {
        Dataset<Row> basDs = loadDataset(BASICS_FILE);
        Dataset<Row> ratDs = loadDataset(RATINGS_FILE);
        Dataset<Row> actDs = loadDataset(PRINCIPALS_FILE);
        Dataset<Row> nameDs = loadDataset(NAMES_FILE);

        Dataset<Row> result = basDs
            .join(ratDs)
                .where(basDs.col("tconst").equalTo(ratDs.col("tconst")))
                .select(basDs.col("tconst"), col("numVotes"), col("titleType"))
                .where(col("titleType").equalTo("movie")
                    .and(col("numVotes").cast(DataTypes.IntegerType).geq(100000)))
                .drop("numVotes", "titleType");

        result = result.join(actDs.where(col("category").equalTo("actor")))
            .where((result.col("tconst").equalTo(actDs.col("tconst"))))
            .select(result.col("tconst"), col("nconst"));

        result = result.join(nameDs)
            .where((result.col("nconst").equalTo(nameDs.col("nconst"))))
            .select(result.col("tconst"), col("primaryName"))
            .groupBy("primaryName").count()
            .where(col("count").geq(10))
            .drop("count");

        saveToFile("src/main/resources/task1/output/top-actors", result);
    }

    /**
     * Find top 5 films for each director
     */
    private static void topDirectorFilms() {
        Dataset<Row> basDs = loadDataset(BASICS_FILE);
        Dataset<Row> ratDs = loadDataset(RATINGS_FILE);
        Dataset<Row> actDs = loadDataset(PRINCIPALS_FILE);
        Dataset<Row> nameDs = loadDataset(NAMES_FILE);

        Dataset<Row> result = basDs.join(ratDs)
            .where(basDs.col("tconst").equalTo(ratDs.col("tconst")))
            .select(col("primaryTitle"), col("startYear"),
                col("averageRating"), col("numVotes"), basDs.col("tconst"))
            .sort(desc("numVotes"));
        result = result.join(actDs.where(col("category").equalTo("director")))
            .where(result.col("tconst").equalTo(actDs.col("tconst")));
        result = result.join(nameDs)
            .where(result.col("nconst").equalTo(nameDs.col("nconst")));

        WindowSpec windowsSpec = Window.partitionBy(col("primaryName"))
            .orderBy(desc("averageRating"));

        result = result.withColumn("row_number", row_number().over(windowsSpec))
            .where(col("row_number").leq(5))
            .select(col("primaryName"), col("primaryTitle"),
                col("startYear"), col("averageRating"), col("numVotes"));

        result.show();
        saveToFile("src/main/resources/task1/output/top-director-films", basDs);
    }

    private static Dataset<Row> loadDataset(String path) {
        return spark.read().option("delimiter", "\t").option("header", true)
            .csv(path);
    }

    private static void saveToFile(String path, Dataset<Row> ds) {
        ds.coalesce(1)
            .write()
            .option("header", "true")
            .mode(SaveMode.Overwrite)
            .csv(path);
    }
}
