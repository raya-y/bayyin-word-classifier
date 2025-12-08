# Quick Start Guide

## Starting the Backend Server

1. **Install dependencies** (if not already installed):
   ```bash
   cd backend
   python -m pip install -r requirements.txt
   ```

2. **Start the server**:
   ```bash
   python -m app.main
   ```

   Or using uvicorn directly:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Verify the server is running**:
   - Open http://localhost:8000/docs in your browser to see the API documentation
   - Or check http://localhost:8000/health to see if models are loaded

## Starting the Frontend

1. **In a separate terminal**, navigate to the project root:
   ```bash
   npm install  # if needed
   npm run dev
   ```

2. **Update API URL** (if backend is on a different port/URL):
   - Edit `src/pages/Index.tsx`
   - Change the `API_URL` constant to match your backend URL

## Testing

1. **Test with Arabic text**:
   - Enter Arabic text in the frontend
   - Should get predictions from all models

2. **Test with English text**:
   - Enter English text
   - Should show an error message in Arabic: "النص المدخل ليس بالعربية"

3. **Check backend logs**:
   - Watch the terminal where the backend is running
   - You should see model loading messages on startup
   - You should see prediction requests when classifying text

## Troubleshooting

### Backend not starting
- Check if port 8000 is already in use
- Verify all dependencies are installed: `python -m pip list`
- Check Python version: `python --version` (should be 3.11+)

### Models not loading
- Check `backend/config.yaml` has correct model repository IDs
- Verify you have internet connection (models download from Hugging Face)
- Check backend logs for specific error messages

### Frontend can't connect to backend
- Verify backend is running on http://localhost:8000
- Check `API_URL` in `src/pages/Index.tsx` matches backend URL
- Check browser console for CORS errors (backend should have CORS enabled)

### Getting different results each time
- This was the old mock function behavior
- Now it should give consistent results from actual models
- If still getting random results, check that frontend is calling the backend (check Network tab in browser DevTools)

