import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import axios from 'axios'
import RegisterPage from '../pages/RegisterPage'
import { register } from '@/api/auth'

// --- module mocks ---

const mockNavigate = vi.fn()
const mockSetAuth = vi.fn()

vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>()
  return { ...actual, useNavigate: () => mockNavigate }
})

vi.mock('@/store/authStore', () => ({
  useAuthStore: () => ({ setAuth: mockSetAuth }),
}))

vi.mock('@/api/auth')

// --- helpers ---

function makeQueryClient() {
  return new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  })
}

function renderPage() {
  render(
    <QueryClientProvider client={makeQueryClient()}>
      <MemoryRouter>
        <RegisterPage />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

function makeAxiosError(detail: string) {
  const err = new axios.AxiosError('Request failed')
  err.response = {
    data: { detail },
    status: 422,
    statusText: 'Unprocessable Entity',
    headers: {},
    config: { headers: {} } as any,
  }
  return err
}

async function fillValidForm(
  user: ReturnType<typeof userEvent.setup>,
  overrides: Partial<{
    fullName: string
    username: string
    email: string
    password: string
    confirmPassword: string
  }> = {}
) {
  const {
    fullName = 'John Doe',
    username = 'johndoe',
    email = 'john@example.com',
    password = 'Password1',
    confirmPassword = 'Password1',
  } = overrides

  await user.type(screen.getByLabelText('Full Name'), fullName)
  await user.type(screen.getByLabelText('Username'), username)
  await user.type(screen.getByLabelText('Email'), email)
  await user.type(screen.getByLabelText('Password'), password)
  await user.type(screen.getByLabelText('Confirm Password'), confirmPassword)
  await user.click(screen.getByLabelText(/I agree to the/i))
}

// --- tests ---

describe('RegisterPage form validation', () => {
  beforeEach(() => {
    vi.mocked(register).mockReset()
    mockNavigate.mockReset()
    mockSetAuth.mockReset()
    vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('shows required field errors when submitting an empty form', async () => {
    const user = userEvent.setup()
    renderPage()

    await user.click(screen.getByRole('button', { name: 'Create Account' }))

    expect(screen.getByText('Full name is required')).toBeInTheDocument()
    expect(screen.getByText('Username is required')).toBeInTheDocument()
    expect(screen.getByText('Email is required')).toBeInTheDocument()
    expect(screen.getByText('Password is required')).toBeInTheDocument()
    expect(screen.getByText('Please confirm your password')).toBeInTheDocument()
    expect(screen.getByText('You must accept the terms to continue')).toBeInTheDocument()
  })

  it('shows inline error when passwords do not match', async () => {
    const user = userEvent.setup()
    renderPage()

    await fillValidForm(user, { confirmPassword: 'Different1' })
    await user.click(screen.getByRole('button', { name: 'Create Account' }))

    expect(screen.getByText('Passwords do not match')).toBeInTheDocument()
  })

  it('shows backend error inline for weak password response', async () => {
    vi.mocked(register).mockRejectedValue(makeAxiosError('Password too weak'))
    const user = userEvent.setup()
    renderPage()

    await fillValidForm(user)
    await user.click(screen.getByRole('button', { name: 'Create Account' }))

    await waitFor(() => {
      expect(screen.getByText('Password too weak')).toBeInTheDocument()
    })
  })

  it('shows backend error inline for duplicate email', async () => {
    vi.mocked(register).mockRejectedValue(makeAxiosError('Email already registered'))
    const user = userEvent.setup()
    renderPage()

    await fillValidForm(user)
    await user.click(screen.getByRole('button', { name: 'Create Account' }))

    await waitFor(() => {
      expect(screen.getByText('Email already registered')).toBeInTheDocument()
    })
  })

  it('clears field errors once the user corrects the invalid input', async () => {
    const user = userEvent.setup()
    renderPage()

    // Trigger validation errors
    await user.click(screen.getByRole('button', { name: 'Create Account' }))
    expect(screen.getByText('Full name is required')).toBeInTheDocument()
    expect(screen.getByText('Password is required')).toBeInTheDocument()

    // Correct full name and password, then re-submit
    await user.type(screen.getByLabelText('Full Name'), 'John Doe')
    await user.type(screen.getByLabelText('Username'), 'johndoe')
    await user.type(screen.getByLabelText('Email'), 'john@example.com')
    await user.type(screen.getByLabelText('Password'), 'Password1')
    await user.type(screen.getByLabelText('Confirm Password'), 'Password1')
    await user.click(screen.getByLabelText(/I agree to the/i))
    await user.click(screen.getByRole('button', { name: 'Create Account' }))

    expect(screen.queryByText('Full name is required')).not.toBeInTheDocument()
    expect(screen.queryByText('Password is required')).not.toBeInTheDocument()
  })

  it('redirects to /dashboard on successful registration', async () => {
    vi.mocked(register).mockResolvedValue({
      user: {
        id: '1',
        email: 'john@example.com',
        username: 'johndoe',
        created_at: '2024-01-01T00:00:00Z',
      },
      access_token: 'fake-token',
      refresh_token: 'fake-refresh',
      token_type: 'bearer',
      expires_in: 3600,
    })
    const user = userEvent.setup()
    renderPage()

    await fillValidForm(user)
    await user.click(screen.getByRole('button', { name: 'Create Account' }))

    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/dashboard')
    })
  })
})
