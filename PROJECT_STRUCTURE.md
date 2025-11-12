# Productivity Tracker - Project Structure

```
productivity-tracker/
├── tracker/                  # Python attention tracking
│   ├── core/
│   ├── utils/
│   ├── config/
│   └── *.py
│
├── backend/                  # FastAPI backend
│   ├── api/routes/
│   ├── models/
│   ├── services/
│   ├── middleware/
│   ├── config/
│   └── utils/
│
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── hooks/
│   │   └── utils/
│   └── public/
│
├── database/                 # Database schemas
│   ├── schemas/
│   ├── migrations/
│   └── seeds/
│
├── scripts/                  # Utility scripts
├── tests/                    # Tests
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
└── .env                      # Environment variables
```

## Tech Stack

**Tracker**: Python, OpenCV, dlib  
**Backend**: FastAPI, Supabase  
**Frontend**: React, TypeScript, Tailwind CSS  
**Database**: Supabase (PostgreSQL)
