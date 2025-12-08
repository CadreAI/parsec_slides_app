import { auth } from '@clerk/nextjs/server'
import { redirect } from 'next/navigation'
import SignInContent from './SignInContent'

export default async function SignInPage() {
    const { userId } = await auth()
    if (userId) {
        redirect('/dashboard')
    }
    return <SignInContent />
}
