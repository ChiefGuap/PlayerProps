const handleSubmit = async (event) => {
    event.preventDefault();
    try {
        const response = await fetch('http://localhost:5000/generate-location', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',  // Set the correct Content-Type
            },
            body: JSON.stringify({ foo, bar, baz }),  // Your data
        });
        // Rest of your code...
    } catch (error) {
        // Handle the error...
    }
};
