// --- Global Variables ---
let model;
const MODEL_URL = './model/model.json'; // Path to the converted TF.js model files

// NOTE: These values MUST match the scaling (mean/std or min/max) used 
// during the Python training phase.
const SCALING_PARAMS = {
    // Example Min-Max scaling parameters for the 4 features:
    min: [0.01, 0.5, 0.00001, 0.001], // Example minimum values from training data
    max: [500, 15, 0.1, 0.5],       // Example maximum values from training data
};

const FEATURES = ['period', 'duration', 'depth', 'radius_ratio'];

// --- Utility Functions ---

/**
 * Normalizes the input data using the pre-calculated min/max from the training set.
 * (x - min) / (max - min)
 * @param {Array<number>} data - The raw input feature values.
 * @returns {tf.Tensor} - The normalized 2D tensor.
 */
function normalizeData(data) {
    const dataTensor = tf.tensor2d([data]);
    
    // Normalize using the global scaling constants
    const min = tf.tensor1d(SCALING_PARAMS.min);
    const max = tf.tensor1d(SCALING_PARAMS.max);

    // Apply normalization: (x - min) / (max - min)
    const normalizedTensor = dataTensor.sub(min).div(max.sub(min));
    
    // Clean up Tensors from memory (important for performance)
    min.dispose();
    max.dispose();

    return normalizedTensor;
}

// --- Main Model Logic ---

/**
 * Loads the pre-trained TensorFlow.js model.
 */
async function loadModel() {
    try {
        document.getElementById('model-status').textContent = 'Model Status: Downloading...';
        model = await tf.loadLayersModel(MODEL_URL);
        document.getElementById('model-status').textContent = 'Model Status: Ready (In-Browser)';
        document.getElementById('predict-button').disabled = false;
        document.getElementById('log-data-button').disabled = false;
    } catch (error) {
        console.error("Error loading model:", error);
        document.getElementById('model-status').textContent = 'Model Status: Error loading model.';
    }
}

/**
 * Runs the prediction when the form is submitted.
 */
async function runPrediction(event) {
    event.preventDefault();
    if (!model) {
        alert("Model is still loading. Please wait.");
        return;
    }

    // 1. Collect user input and convert to an array of numbers
    const rawData = FEATURES.map(id => parseFloat(document.getElementById(id).value));
    
    // 2. Pre-process the data (Normalization)
    const inputTensor = normalizeData(rawData);

    // 3. Run the prediction
    const prediction = model.predict(inputTensor);
    const confidence = (await prediction.data())[0]; // Get the single confidence score

    // 4. Determine the classification
    const threshold = 0.5;
    const isPlanet = confidence >= threshold;
    const classificationText = isPlanet ? 'LIKELY EXOPLANET' : 'LIKELY FALSE POSITIVE';
    
    // 5. Update the UI
    document.getElementById('prediction-text').textContent = classificationText;
    document.getElementById('confidence-score').textContent = `${(confidence * 100).toFixed(2)}% (Planet)`;
    document.getElementById('prediction-text').style.color = isPlanet ? 'var(--confirmed-green)' : 'var(--false-red)';
    
    // 6. Update the visualization
    updateChart(rawData, isPlanet);

    // Clean up tensor memory
    inputTensor.dispose();
    prediction.dispose();
}

// --- Data Visualization (Chart.js) ---
let exoplanetChart;
const PRE_LOADED_DATA = [
    // A VERY small, curated subset of the NASA data (Period vs Radius Ratio) 
    // to give visual context for the model. This array is pre-saved in the JS file.
    { x: 1.5, y: 0.05, label: 'Confirmed' },
    { x: 10, y: 0.02, label: 'Confirmed' },
    { x: 50, y: 0.005, label: 'Confirmed' },
    { x: 0.7, y: 0.1, label: 'False Positive' },
    { x: 5, y: 0.01, label: 'False Positive' },
];
let userPoint = null;

function initializeChart() {
    const ctx = document.getElementById('exoplanet-chart').getContext('2d');
    
    const confirmedData = PRE_LOADED_DATA.filter(d => d.label === 'Confirmed');
    const falsePositiveData = PRE_LOADED_DATA.filter(d => d.label === 'False Positive');

    exoplanetChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Confirmed Exoplanets (Sample)',
                    data: confirmedData,
                    backgroundColor: 'rgba(61, 153, 112, 0.7)', // Confirmed Green
                },
                {
                    label: 'False Positives (Sample)',
                    data: falsePositiveData,
                    backgroundColor: 'rgba(255, 65, 54, 0.7)', // False Red
                },
                {
                    label: 'User Input',
                    data: userPoint ? [userPoint] : [],
                    backgroundColor: 'var(--star-yellow)',
                    pointRadius: 10,
                    pointStyle: 'star',
                }
            ]
        },
        options: {
            // ... Chart Options for axes labels, dark theme, etc.
            scales: {
                x: {
                    type: 'logarithmic', // Use log scale for Period (X-axis)
                    title: { display: true, text: 'Orbital Period (Days, Log Scale)', color: '#fff' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#fff' }
                },
                y: {
                    title: { display: true, text: 'Planet-to-Star Radius Ratio', color: '#fff' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#fff' }
                }
            },
            plugins: { legend: { labels: { color: '#fff' } } }
        }
    });
}

function updateChart(rawData, isPlanet) {
    // We plot Period vs. Radius Ratio (Indices 0 and 3 in the rawData array)
    userPoint = { 
        x: rawData[0], 
        y: rawData[3], 
        label: isPlanet ? 'AI Prediction: Planet' : 'AI Prediction: False Positive' 
    };
    
    exoplanetChart.data.datasets[2].data = [userPoint];
    exoplanetChart.update();
}


// --- Citizen Science / Data Logging ---

/**
 * Logs the current user data and prediction to the browser's local storage.
 */
function logDataToLocalStorage() {
    const rawData = FEATURES.map(id => parseFloat(document.getElementById(id).value));
    const prediction = document.getElementById('prediction-text').textContent;

    if (!prediction || prediction === '---') {
        alert("Please run a classification first.");
        return;
    }

    const logEntry = {
        period: rawData[0],
        duration: rawData[1],
        depth: rawData[2],
        radius_ratio: rawData[3],
        ai_prediction: prediction,
        timestamp: new Date().toISOString()
    };

    let log = JSON.parse(localStorage.getItem('exoplanetLog') || '[]');
    log.push(logEntry);
    localStorage.setItem('exoplanetLog', JSON.stringify(log));

    document.getElementById('log-count').textContent = `Logged points: ${log.length}`;
    alert("Data point logged! Thank you for your contribution to the citizen science dataset.");

    // Note: A real implementation would export this data via a download button 
    // for later model retraining by the developers.
}


// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    // 1. Load the ML Model
    loadModel();

    // 2. Initialize the visualization chart
    initializeChart();

    // 3. Attach event listeners
    document.getElementById('prediction-form').addEventListener('submit', runPrediction);
    document.getElementById('log-data-button').addEventListener('click', logDataToLocalStorage);
});