async function getPrediction(inputData) {
    try {
        const response = await fetch('http://localhost:3000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ data: inputData }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Error fetching prediction:', error);
    }
}

// Example usage
const inputData = [/* your input data here */];
getPrediction(inputData).then(result => {
    console.log('Prediction result:', result);
});

document.getElementsByClassName('predictButton').addEventListener('click', () => {
    const inputData = [/* collect input data from the user */];
    getPrediction(inputData).then(result => {
        console.log('Prediction result:', result);
    });
});
