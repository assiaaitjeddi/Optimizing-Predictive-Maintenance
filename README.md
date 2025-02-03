<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimizing Predictive Maintenance</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            line-height: 1.6;
            background-color: #f4f4f4;
            color: #333;
        }
        .container {
            max-width: 900px;
            margin: 20px auto;
            padding: 20px;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3 {
            color: #0056b3;
        }
        a {
            color: #007bff;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .authors {
            margin-top: 10px;
            padding: 10px;
            background: #f9f9f9;
            border-left: 4px solid #0056b3;
        }
        .section {
            margin-bottom: 20px;
        }
        .section img {
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        .keywords {
            background: #e9ecef;
            padding: 10px;
            border-radius: 5px;
        }
        .limitations {
            background: #ffe5e5;
            padding: 10px;
            border-left: 4px solid #d9534f;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Optimizing Predictive Maintenance Strategies</h1>
        <h2>Using Reinforcement Learning and RUL Estimation</h2>
        
        <div class="authors">
            <p><strong>Chaimae Elfakir</strong><br>
            Dept. of Mathematics and Computer Science<br>
            University Hassan II, ENSET Mohammedia<br>
            Mohammedia, Morocco<br>
            <a href="mailto:chaimae.elfakir-etu@etu.univh2c.ma">chaimae.elfakir-etu@etu.univh2c.ma</a></p>
            
            <p><strong>Assia Ait Jeddi</strong><br>
            Dept. of Mathematics and Computer Science<br>
            University Hassan II, ENSET Mohammedia<br>
            Mohammedia, Morocco<br>
            <a href="mailto:assia.aitjeddi-etu@univh2c.ma">assia.aitjeddi-etu@univh2c.ma</a></p>
        </div>

        <div class="section">
            <h2>Keywords</h2>
            <div class="keywords">
                Predictive Maintenance, Reinforcement Learning, RUL Estimation, Spatio-Temporal Analysis, Machine Learning
            </div>
        </div>

        <div class="section">
            <h2>Abstract</h2>
            <p>This project investigates strategies for optimizing predictive maintenance using reinforcement learning techniques and Remaining Useful Life (RUL) estimation. Through the analysis of datasets such as NASA C-MAPSS and MetroPT-3, the study explores advanced methodologies to improve maintenance scheduling and system reliability.</p>
        </div>

        <div class="section">
            <h2>Datasets</h2>
            <p><strong>MetroPT-3 Dataset:</strong> Developed for monitoring pneumatic and mechanical systems in trains. Features include:</p>
            <ul>
                <li>Detailed sensor measurements (e.g., pressures, temperatures).</li>
                <li>GPS data for spatio-temporal analysis.</li>
                <li>Tools for multi-anomaly detection.</li>
            </ul>

            <p><strong>NASA C-MAPSS Dataset:</strong> Simulated "run-to-failure" data for aircraft engines, focusing on RUL prediction.</p>
        </div>

        <div class="section">
            <h2>Results</h2>
            <p>Short-term predictions using XGBoost showed near-perfect accuracy within 24-hour windows. However, accuracy declined significantly for longer horizons (36-48 hours).</p>
            <img src="path-to-image" alt="Prediction Results Graph">
        </div>

        <div class="section limitations">
            <h3>Limitations</h3>
            <p>The dataset's design strongly favors 24-hour predictions, making it less suitable for long-term forecasts. Simultaneous failures and complex sensor interactions require further analysis for robust real-world applications.</p>
        </div>
    </div>
</body>
</html>
